"""Tests for startup key validation (validate_startup_keys)."""

from __future__ import annotations

import os

import pytest


def test_validate_passes_when_key_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
    monkeypatch.delenv("CLORIS_SKIP_STARTUP_VALIDATION", raising=False)
    from shared.config import validate_startup_keys

    validate_startup_keys()  # must not raise


def test_validate_raises_when_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLORIS_SKIP_STARTUP_VALIDATION", raising=False)
    from shared.config import MissingRequiredKeyError, validate_startup_keys

    with pytest.raises(MissingRequiredKeyError) as exc_info:
        validate_startup_keys()

    assert "ANTHROPIC_API_KEY" in exc_info.value.missing
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)


def test_validate_skipped_by_env_var(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("CLORIS_SKIP_STARTUP_VALIDATION", "1")
    from shared.config import validate_startup_keys

    validate_startup_keys()  # must not raise despite missing key


def test_validate_skip_flag_variants(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from shared.config import validate_startup_keys

    for value in ("1", "true", "True", "TRUE", "yes", "on"):
        monkeypatch.setenv("CLORIS_SKIP_STARTUP_VALIDATION", value)
        validate_startup_keys()  # must not raise


def test_missing_required_key_error_attributes():
    from shared.config import MissingRequiredKeyError

    err = MissingRequiredKeyError(["ANTHROPIC_API_KEY", "OPENAI_API_KEY"])
    assert err.missing == ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]
    assert "ANTHROPIC_API_KEY" in str(err)
    assert "OPENAI_API_KEY" in str(err)
