"""Tests for BearerAuthMiddleware and the /api/bootstrap endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def strict_client(monkeypatch):
    """TestClient with auth enforcement active (CLORIS_SKIP_AUTH_FOR_TESTING cleared)."""
    monkeypatch.delenv("CLORIS_SKIP_AUTH_FOR_TESTING", raising=False)
    from cloris.app import create_app
    from cloris.api.auth import SESSION_TOKEN

    return TestClient(create_app(), raise_server_exceptions=False), SESSION_TOKEN


def test_healthz_exempt_from_auth(strict_client):
    client, _ = strict_client
    response = client.get("/healthz")
    assert response.status_code == 200


def test_root_exempt_from_auth(strict_client):
    client, _ = strict_client
    response = client.get("/")
    # / serves the built Svelte index.html — may be 200 or 404 depending on
    # whether the frontend is built; either way, not a 401.
    assert response.status_code != 401


def test_bootstrap_exempt_from_auth(strict_client):
    client, _ = strict_client
    response = client.get("/api/bootstrap")
    assert response.status_code == 200
    body = response.json()
    assert "token" in body
    assert isinstance(body["token"], str)
    assert len(body["token"]) == 64  # 32-byte hex = 64 chars


def test_authenticated_endpoint_missing_token(strict_client):
    client, _ = strict_client
    response = client.get("/api/status")
    assert response.status_code == 401
    assert response.json()["detail"] == "missing_bearer_token"


def test_authenticated_endpoint_wrong_token(strict_client):
    client, _ = strict_client
    response = client.get("/api/status", headers={"Authorization": "Bearer wrong-token"})
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_bearer_token"


def test_authenticated_endpoint_correct_token(strict_client):
    client, token = strict_client
    # /api/onboarding/status is a simple read endpoint with no external deps.
    response = client.get(
        "/api/onboarding/status",
        headers={"Authorization": f"Bearer {token}"},
    )
    # 200 or any non-401 status means auth passed.
    assert response.status_code != 401


def test_sse_stream_requires_token_query_param(strict_client, monkeypatch):
    client, token = strict_client
    # Without token: 401.
    response = client.get("/api/conversation/test-brief/stream")
    assert response.status_code == 401
    # Disable the infinite SSE body so the test can assert that query-token
    # auth passes through to the route layer without trying to consume a
    # never-ending stream.
    monkeypatch.setenv("CLORIS_CONVERSATION_SSE_DISABLED", "1")
    response = client.get(f"/api/conversation/test-brief/stream?token={token}")
    assert response.status_code == 404


def test_bearer_missing_value(strict_client):
    """'Authorization: Bearer ' with empty value must return 401, not crash."""
    client, _ = strict_client
    response = client.get("/api/status", headers={"Authorization": "Bearer "})
    assert response.status_code == 401


# Slice 1 — /brand/* and /manifest.webmanifest exempt prefixes.
# Browsers fetch these from <link rel="icon"> / <link rel="manifest"> /
# og:image without an Authorization header; the middleware must let them
# through. Status is content-driven (200 when dist/ is built, 404 when
# the test environment hasn't built it), but it must not be 401.
def test_brand_prefix_exempt_from_auth(strict_client):
    client, _ = strict_client
    response = client.get("/brand/cloris-icon.svg")
    assert response.status_code != 401


def test_manifest_exempt_from_auth(strict_client):
    client, _ = strict_client
    response = client.get("/manifest.webmanifest")
    assert response.status_code != 401


# Slice 2 — POST /api/stop/<source>/<state_key> must require a Bearer.
# Without auth the frontend was previously silently 401-ing every stop
# attempt under strict-auth. The route is auth-gated like every other
# mutation; the helper at cloris/frontend/src/lib/api.ts:stopWorker now
# injects Authorization so this 401 stops happening in the UI.
def test_stop_requires_bearer(strict_client):
    client, _ = strict_client
    response = client.post("/api/stop/linkedin/some-state-key")
    assert response.status_code == 401
    assert response.json()["detail"] == "missing_bearer_token"


def test_stop_accepts_valid_bearer(strict_client):
    """With a valid bearer the route progresses past auth.

    The actual handler may return 404 (state dir not found) or another
    application-level status — we only care that auth itself does not
    reject the request.
    """
    client, token = strict_client
    response = client.post(
        "/api/stop/linkedin/some-state-key",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code != 401


# ---------------------------------------------------------------------------
# Audit finding F-5: SSE query-token allowlist is closed-enumeration.
# A future ``/api/conversation/<brief>/admin/stream`` route must NOT
# auto-exempt simply because it shares the prefix and ``/stream``
# suffix; the middleware now matches ``re.fullmatch`` against an
# explicit pattern list.
# ---------------------------------------------------------------------------


def test_intake_converse_stream_accepts_query_token(strict_client):
    client, token = strict_client
    response = client.post(
        f"/api/intake/sessions/9999999/converse/stream?token={token}",
        json={"recruiter_message": "hi"},
    )
    # Auth passed; the application layer handles the unknown session id
    # via an SSE ``error`` event in the body, NOT a 401.
    assert response.status_code != 401


def test_intake_converse_stream_rejects_missing_token(strict_client):
    client, _ = strict_client
    response = client.post(
        "/api/intake/sessions/9999999/converse/stream",
        json={"recruiter_message": "hi"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "invalid_token"


def test_synthetic_nested_conversation_stream_is_not_exempt(strict_client):
    """A path that shares the prefix AND ends in ``/stream`` but is not
    in the explicit allowlist must still require a bearer header.

    Pre-fix this exact shape ``/api/conversation/<brief>/admin/stream``
    would have been query-token-exempt simply because it matched
    ``startswith("/api/conversation/")`` and ``endswith("/stream")``.
    """
    client, token = strict_client
    # Token-as-query-param must NOT be honored on a non-allowlisted path.
    response = client.get(
        f"/api/conversation/some-brief/admin/stream?token={token}"
    )
    # Without a Bearer header the middleware returns 401 with the
    # ``missing_bearer_token`` body; if the synthetic path had been
    # exempt by accident, the middleware would have honored the query
    # token and returned 404 (no such route) instead.
    assert response.status_code == 401
    assert response.json()["detail"] == "missing_bearer_token"


def test_synthetic_intake_nested_stream_is_not_exempt(strict_client):
    """Same shape audit for the conversational intake exemption: nested
    paths under ``/api/intake/sessions/.../converse/`` must not match
    just because they end in ``/stream``."""
    client, token = strict_client
    response = client.post(
        f"/api/intake/sessions/1/converse/admin/stream?token={token}",
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "missing_bearer_token"


def test_sse_query_token_allowlist_unit_match():
    """Direct unit guard on the helper. The middleware uses this same
    helper, so a regex regression here is a visible test failure."""

    from cloris.api.auth import _is_sse_query_token_route

    assert _is_sse_query_token_route("/api/conversation/abc/stream") is True
    assert _is_sse_query_token_route(
        "/api/intake/sessions/42/converse/stream"
    ) is True
    # Negative cases — extra path segments must not match.
    assert _is_sse_query_token_route(
        "/api/conversation/abc/admin/stream"
    ) is False
    assert _is_sse_query_token_route(
        "/api/intake/sessions/42/converse/admin/stream"
    ) is False
    # Adjacent (non-conversation) routes must not match.
    assert _is_sse_query_token_route("/api/run/linkedin/abc/stream") is False
    assert _is_sse_query_token_route("/api/status") is False
