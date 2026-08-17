"""Pytest configuration for the Cloris test suite.

Sets environment variables that must be in place before any Cloris module
is imported by the test runner.
"""

from __future__ import annotations

import os
import tempfile

# Skip startup key validation in tests — live API keys are not available in CI.
os.environ.setdefault("CLORIS_SKIP_STARTUP_VALIDATION", "1")

# Bypass BearerAuthMiddleware for existing tests that use
# TestClient(create_app()) without auth headers. New tests that specifically
# exercise auth behavior clear or unset this env var via monkeypatch.
os.environ.setdefault("CLORIS_SKIP_AUTH_FOR_TESTING", "1")

# Shadow-judge hermeticity: the operator's .env may carry
# SHADOW_FACIAL_MODEL_ENABLED=true for real runs, but tests must NEVER fire
# live Fireworks calls or depend on the operator's experiment state. Plain
# assignment (not setdefault) — dotenv loads with override=False, so a
# process-env value set before any Cloris import always wins. Shadow tests
# that exercise the flag flip it via monkeypatch on shared.config.
os.environ["SHADOW_FACIAL_MODEL_ENABLED"] = "false"
# Same hermeticity for the shadow STRATEGIST (item 19): the operator's
# .env carries SHADOW_STRATEGY_ENABLED=true for the live experiment; an
# unpinned suite would dispatch real Fable calls from any test that runs
# form_strategy/preflight with a shadow_dir, and would spin the module
# executor that test_strategy_shadow's never-spun-up assertion pins.
os.environ["SHADOW_STRATEGY_ENABLED"] = "false"


# P4.4: point the writable output/ tree at an ephemeral per-session tmp root
# instead of the real PROJECT_ROOT/output/.
#
# Most tests already isolate their own state via pytest's `tmp_path` +
# monkeypatching STATE_ROOT/OUTPUT_ROOT. But a handful of tests (e.g.
# tests/test_linkedin_session_orchestrator.py's `_resume_has_pending_work`
# tests) call resolve_linkedin_state_dir()-family helpers without an
# explicit `state_dir=` override, so shared/output_paths.py derives the
# on-disk key from brief content and writes for real under
# shared.config.OUTPUT_DIR — the module-level default, which resolves to
# PROJECT_ROOT/output/ unless overridden. That's the confirmed origin of the
# 74 "*_tmp*"-named debris dirs found under output/state/linkedin/ (built
# from `f"<status>-{Path(tempfile.TemporaryDirectory()).name}"` project ids,
# which slugify_output_component() turns into e.g. "exhausted_tmpXXXXXXXX").
#
# shared/user_data_dir.py already exposes CLORIS_USER_DATA_DIR as the
# override seam (`_env_override_dir()` / `should_use_user_data_dir()`):
# when set, shared.config.OUTPUT_DIR resolves under
# `<CLORIS_USER_DATA_DIR>/output` instead of PROJECT_ROOT/output. Setting it
# here — before any Cloris module is imported, same discipline as the two
# os.environ.setdefault calls above — redirects every test in the session
# that forgets an explicit override into throwaway space. This must be a
# session-lifetime tmp dir (not a per-test tmp_path) because
# shared.config.OUTPUT_DIR / shared.output_paths.OUTPUT_ROOT are resolved
# once at import time, not re-resolved per test.
#
# Plain assignment, NOT setdefault: a developer who has CLORIS_USER_DATA_DIR
# exported (the documented power-user opt-in) must never receive test debris
# in their real user-data dir — tests always isolate, unconditionally.
os.environ["CLORIS_USER_DATA_DIR"] = tempfile.mkdtemp(
    prefix="cloris-pytest-userdata-"
)


import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def clear_llm_client_cache():
    """Keep process-global SDK client reuse from leaking across tests."""
    from shared.llm_clients import get_llm_client

    get_llm_client.cache_clear()
    yield
    get_llm_client.cache_clear()


@pytest.fixture()
def auth_client():
    """TestClient with the session bearer token pre-loaded in default headers.

    Use this fixture in tests that explicitly exercise auth-gated endpoints
    without the CLORIS_SKIP_AUTH_FOR_TESTING bypass active.
    """
    from cloris.app import create_app
    from cloris.api.auth import SESSION_TOKEN

    return TestClient(
        create_app(),
        headers={"Authorization": f"Bearer {SESSION_TOKEN}"},
    )
