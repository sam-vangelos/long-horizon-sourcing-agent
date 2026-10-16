"""Which Anthropic account pays — ANTHROPIC_API_KEY_SOURCE.

The switch lets the personal key carry a campaign WITHOUT a key paste and
WITHOUT any silent fallback between accounts. Every test here reloads
``shared.config`` under a patched environment and reloads it clean again in
``finally`` so later tests in the same process see the real values.

``AGENT_KEY_PREFIX`` is read at import into ``config._AGENT_PREFIX``; the live
LinkedIn entry point sets it, so each test sets or clears it EXPLICITLY rather
than inheriting whatever the test process happens to carry.
"""

from __future__ import annotations

import importlib

import pytest

import shared.config as shared_config


def _reload_clean() -> None:
    """Restore the module to the real environment (call after monkeypatch.undo)."""

    importlib.reload(shared_config)


def test_company_source_is_default_and_keeps_agent_prefix_precedence(monkeypatch):
    """Unset knob == today's behavior: _agent_key, prefix wins over the shared key."""

    monkeypatch.delenv("ANTHROPIC_API_KEY_SOURCE", raising=False)
    monkeypatch.setenv("AGENT_KEY_PREFIX", "LINKEDIN")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "shared-company-key")
    monkeypatch.setenv("LINKEDIN_ANTHROPIC_API_KEY", "prefixed-company-key")
    monkeypatch.setenv("PERSONAL_ANTHROPIC_API_KEY", "personal-key")
    try:
        reloaded = importlib.reload(shared_config)
        assert reloaded.ANTHROPIC_API_KEY_SOURCE == "company"
        assert reloaded.ANTHROPIC_API_KEY == "prefixed-company-key"
        assert reloaded.anthropic_key_fingerprint() == "company:…-key"
    finally:
        monkeypatch.undo()
        _reload_clean()


def test_personal_source_reads_personal_key_and_ignores_company_and_prefixed_keys(
    monkeypatch,
):
    """personal == PERSONAL_ANTHROPIC_API_KEY, read flat. No prefix, no fallback."""

    monkeypatch.setenv("ANTHROPIC_API_KEY_SOURCE", "personal")
    monkeypatch.setenv("AGENT_KEY_PREFIX", "LINKEDIN")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "shared-company-key")
    monkeypatch.setenv("LINKEDIN_ANTHROPIC_API_KEY", "prefixed-company-key")
    # An agent-prefixed PERSONAL key must NOT be consulted either: the whole
    # point of the flat read is that one variable names the payer.
    monkeypatch.setenv("LINKEDIN_PERSONAL_ANTHROPIC_API_KEY", "prefixed-personal-key")
    monkeypatch.setenv("PERSONAL_ANTHROPIC_API_KEY", "the-personal-key")
    try:
        reloaded = importlib.reload(shared_config)
        assert reloaded.ANTHROPIC_API_KEY_SOURCE == "personal"
        assert reloaded.ANTHROPIC_API_KEY == "the-personal-key"
        assert reloaded.ANTHROPIC_API_KEY != "shared-company-key"
        assert reloaded.ANTHROPIC_API_KEY != "prefixed-company-key"
        assert reloaded.ANTHROPIC_API_KEY != "prefixed-personal-key"
        # The company keys are still perfectly readable from the environment —
        # the source knob, not their absence, is what keeps them out.
        assert reloaded.OPENAI_API_KEY is not None
        assert reloaded.anthropic_key_fingerprint() == "personal:…-key"
    finally:
        monkeypatch.undo()
        _reload_clean()


def test_personal_source_with_empty_personal_key_does_not_fall_back(monkeypatch):
    """An empty personal key resolves EMPTY and fails startup — never the company key."""

    monkeypatch.setenv("ANTHROPIC_API_KEY_SOURCE", "personal")
    monkeypatch.delenv("AGENT_KEY_PREFIX", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "company-key-still-present")
    monkeypatch.setenv("PERSONAL_ANTHROPIC_API_KEY", "")
    monkeypatch.delenv("CLORIS_SKIP_STARTUP_VALIDATION", raising=False)
    try:
        reloaded = importlib.reload(shared_config)
        assert reloaded.ANTHROPIC_API_KEY == ""
        assert reloaded.anthropic_key_fingerprint() == "personal:MISSING"

        with pytest.raises(reloaded.MissingRequiredKeyError) as exc_info:
            reloaded.validate_startup_keys()

        message = str(exc_info.value)
        assert (
            "ANTHROPIC_API_KEY_SOURCE=personal but "
            "PERSONAL_ANTHROPIC_API_KEY is empty" in message
        )
        assert exc_info.value.missing == ["PERSONAL_ANTHROPIC_API_KEY"]
        # The dead company value must not appear anywhere in the diagnosis.
        assert "company-key-still-present" not in message
    finally:
        monkeypatch.undo()
        _reload_clean()


def test_unknown_source_raises_at_import(monkeypatch):
    """A typo'd source is a hard stop, not a silent default to either account."""

    monkeypatch.setenv("ANTHROPIC_API_KEY_SOURCE", "corporate")
    monkeypatch.delenv("AGENT_KEY_PREFIX", raising=False)
    try:
        with pytest.raises(ValueError) as exc_info:
            importlib.reload(shared_config)
        message = str(exc_info.value)
        assert "ANTHROPIC_API_KEY_SOURCE" in message
        assert "company" in message and "personal" in message
    finally:
        # The failed reload left the module half-executed; this restores it.
        monkeypatch.undo()
        _reload_clean()
    assert shared_config.ANTHROPIC_API_KEY_SOURCE in {"company", "personal"}
    assert hasattr(shared_config, "OUTPUT_DIR")


def test_fingerprint_never_exposes_more_than_last_four(monkeypatch):
    """The fingerprint is a receipt, not a key: source + at most four characters."""

    secret = "sk-ant-api03-DO-NOT-PRINT-ME-EVER-WXYZ"
    monkeypatch.setenv("ANTHROPIC_API_KEY_SOURCE", "personal")
    monkeypatch.delenv("AGENT_KEY_PREFIX", raising=False)
    monkeypatch.setenv("PERSONAL_ANTHROPIC_API_KEY", secret)
    try:
        reloaded = importlib.reload(shared_config)
        fingerprint = reloaded.anthropic_key_fingerprint()
        assert fingerprint == "personal:…WXYZ"
        assert secret not in fingerprint
        assert secret[:-4] not in fingerprint
        # Nothing but source, separator, ellipsis and four characters.
        assert len(fingerprint) == len("personal:") + 1 + 4
        # Count how much of the key leaked: the longest suffix present must be 4.
        longest = max(
            (n for n in range(len(secret) + 1) if secret[len(secret) - n:] in fingerprint),
            default=0,
        )
        assert longest == 4
    finally:
        monkeypatch.undo()
        _reload_clean()

    # And the empty case says MISSING rather than rendering an empty suffix.
    monkeypatch.setenv("ANTHROPIC_API_KEY_SOURCE", "personal")
    monkeypatch.delenv("AGENT_KEY_PREFIX", raising=False)
    monkeypatch.setenv("PERSONAL_ANTHROPIC_API_KEY", "")
    try:
        reloaded = importlib.reload(shared_config)
        assert reloaded.anthropic_key_fingerprint() == "personal:MISSING"
    finally:
        monkeypatch.undo()
        _reload_clean()
