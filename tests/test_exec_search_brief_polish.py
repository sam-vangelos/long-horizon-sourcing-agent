"""Tests for Executive Search Slice 8 — brief polish exec-register.

Pins:

- :func:`_confidentiality_class_drift` returns ``None`` for OPEN /
  empty seeds, returns a drift descriptor for downgrades / changes
  on REFERENCEABLE / BLIND seeds. Slice 6's named-cascade-route
  discipline.
- :func:`build_brief_polish_exec_system_prompt` returns the base
  prompt + the executive-register addendum, with the addendum
  carrying the confidentiality / prior_search / executive_calibration
  preservation contracts.
- :func:`build_brief_polish_system_prompt` (non-exec base prompt)
  is unchanged — the exec register is additive, not destructive
  (characterization regression).
- :func:`_is_exec_search_brief` distinguishes exec from classic
  briefs by ``target_modules``.
"""

from __future__ import annotations

import pytest

from market_intelligence.brief_polish import (
    _confidentiality_class_drift,
    _is_exec_search_brief,
    build_brief_polish_exec_system_prompt,
    build_brief_polish_system_prompt,
)


# ---------------------------------------------------------------------------
# _confidentiality_class_drift
# ---------------------------------------------------------------------------


def test_drift_none_when_seeded_has_no_class() -> None:
    assert _confidentiality_class_drift(seeded={}, polished={}) is None


@pytest.mark.parametrize("seed_value", ["", "open"])
def test_drift_none_when_seeded_is_open_or_empty(seed_value: str) -> None:
    """Open / empty are not load-bearing — polish may freely drop or
    add (it'll default to "open" downstream)."""

    seeded = {"confidentiality_class": seed_value}
    polished = {}
    assert _confidentiality_class_drift(seeded=seeded, polished=polished) is None


def test_drift_returns_dropped_when_referenceable_seed_drops() -> None:
    seeded = {"confidentiality_class": "referenceable"}
    polished = {"role_title": "x"}
    drift = _confidentiality_class_drift(seeded=seeded, polished=polished)
    assert drift is not None
    assert "dropped" in drift
    assert "referenceable" in drift


def test_drift_returns_changed_when_blind_seed_downgrades_to_open() -> None:
    seeded = {"confidentiality_class": "blind"}
    polished = {"confidentiality_class": "open"}
    drift = _confidentiality_class_drift(seeded=seeded, polished=polished)
    assert drift is not None
    assert "changed" in drift
    assert "blind" in drift
    assert "open" in drift


def test_drift_none_when_seed_and_polish_match_case_insensitive() -> None:
    seeded = {"confidentiality_class": "blind"}
    polished = {"confidentiality_class": "BLIND"}
    assert _confidentiality_class_drift(seeded=seeded, polished=polished) is None


def test_drift_returns_dropped_when_polish_value_is_non_string() -> None:
    seeded = {"confidentiality_class": "blind"}
    polished = {"confidentiality_class": 123}  # type: ignore[dict-item]
    drift = _confidentiality_class_drift(seeded=seeded, polished=polished)
    assert drift is not None
    assert "dropped" in drift


# ---------------------------------------------------------------------------
# Exec-register prompt
# ---------------------------------------------------------------------------


def test_exec_prompt_contains_base_prompt() -> None:
    """The exec prompt is additive — it includes the base prompt
    verbatim so the schema, voice rules, and preservation contracts
    don't drift between exec and non-exec briefs."""

    base = build_brief_polish_system_prompt()
    exec_prompt = build_brief_polish_exec_system_prompt()
    assert exec_prompt.startswith(base)


def test_exec_prompt_adds_executive_register_addendum() -> None:
    exec_prompt = build_brief_polish_exec_system_prompt()
    assert "EXECUTIVE-REGISTER ADDENDUM" in exec_prompt


def test_exec_prompt_carries_confidentiality_preservation_contract() -> None:
    """The addendum names confidentiality_class as a HARD CONTRACT."""

    exec_prompt = build_brief_polish_exec_system_prompt()
    assert "confidentiality_class" in exec_prompt
    assert "HARD CONTRACT" in exec_prompt


def test_exec_prompt_carries_prior_search_preservation_contract() -> None:
    exec_prompt = build_brief_polish_exec_system_prompt()
    assert "prior_search" in exec_prompt
    assert "ruled_out_urls" in exec_prompt


def test_exec_prompt_carries_executive_calibration_preservation_contract() -> None:
    exec_prompt = build_brief_polish_exec_system_prompt()
    assert "executive_calibration" in exec_prompt


def test_base_prompt_does_not_include_exec_addendum() -> None:
    """Characterization regression: classic briefs hit a prompt that
    is byte-identical to the pre-Slice-8 base."""

    base = build_brief_polish_system_prompt()
    assert "EXECUTIVE-REGISTER ADDENDUM" not in base
    # Spot-check: the exec-specific preservation contract for
    # confidentiality_class doesn't leak into the base.
    assert "confidentiality_class" not in base


# ---------------------------------------------------------------------------
# _is_exec_search_brief
# ---------------------------------------------------------------------------


def test_is_exec_search_brief_detects_membership() -> None:
    assert _is_exec_search_brief({"target_modules": ["linkedin", "exec_search"]})
    assert _is_exec_search_brief({"target_modules": ["exec_search"]})


def test_is_exec_search_brief_returns_false_for_classic() -> None:
    assert _is_exec_search_brief({"target_modules": ["linkedin"]}) is False
    assert _is_exec_search_brief({"target_modules": []}) is False
    assert _is_exec_search_brief({}) is False


def test_is_exec_search_brief_handles_malformed_input() -> None:
    assert _is_exec_search_brief({"target_modules": "not a list"}) is False
    assert _is_exec_search_brief("not a dict") is False  # type: ignore[arg-type]
