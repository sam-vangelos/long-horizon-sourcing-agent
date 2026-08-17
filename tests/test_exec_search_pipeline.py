"""Tests for Executive Search Slice 10 — prior-search exclusion + e2e characterization.

Pins:

- :meth:`_merge_prior_search_exclusion` (the helper Slice 10 adds
  to ``linkedin/orchestrator.py``) merges
  ``brief.prior_search.ruled_out_urls`` into ``_seen_urls`` at
  session-init time. Defensive against missing / malformed fields
  (classic briefs without the schema extension stay unaffected).
- The merge is idempotent — re-running yields the same set.
- A V2 brief carrying ``prior_search.ruled_out_urls`` round-trips
  through ``load_brief`` and into the orchestrator's exclusion
  surface.
- End-to-end characterization: a brief with ``target_modules:
  ["linkedin", "exec_search"]`` + 2 prior-search exclusion URLs +
  ``confidentiality_class: "blind"`` + ``executive_calibration``
  populated loads cleanly and exposes the contract every Slice 1-9
  feature reads from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from shared.brief_loader import load_brief
from shared.brief_schema import PriorSearchContext


def _v2_brief_payload(
    *,
    ruled_out_urls: list[str] | None = None,
    target_modules: list[str] | None = None,
) -> dict:
    return {
        "role_title": "VP Engineering",
        "role_summary": "Owns engineering org for a series-D company.",
        "geography": "United States",
        "linkedin_project": "exec",
        "minimum_years_experience": 12,
        "minimum_bar_description": "12+ years engineering leadership.",
        "capability_areas": [
            {
                "name": "Engineering org leadership",
                "description": "Builds and runs 50+ person orgs.",
                "builder_signals": ["VP-level scope"],
                "user_signals": ["IC-level work"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns engineering strategy + delivery.",
            "user_definition": "Manages individual teams without org-wide scope.",
            "edge_case_guidance": "Borderline = full eval.",
        },
        "target_modules": list(target_modules or []),
        "confidentiality_class": "blind",
        "executive_calibration": {
            "sector": "Healthcare",
            "stage": "Series D",
            "pnl_scale_usd": "$200M ARR",
            "register_notes": "Operator-builder bias.",
        },
        "prior_search": {
            "ruled_out_urls": list(ruled_out_urls or []),
        },
        "facial_calibration": {
            "expected_yes_rate_low": 0.3,
            "expected_yes_rate_high": 0.5,
            "fast_exit_patterns": [],
            "trajectory_yes_patterns": [],
            "trajectory_ambiguous_patterns": [],
            "trajectory_no_patterns": [],
        },
    }


def _write_brief(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "brief.json"
    path.write_text(json.dumps(payload))
    return path


# ---------------------------------------------------------------------------
# _merge_prior_search_exclusion — unit (no orchestrator construction)
# ---------------------------------------------------------------------------


class _StubOrchestrator:
    """Minimal stub carrying just the surface
    :meth:`_merge_prior_search_exclusion` reads.

    Slice 10's helper is the load-bearing piece; the rest of the
    orchestrator's init path is brittle to test in isolation.
    Mirrors the surface contract: ``self.brief`` (compat Brief) +
    ``self._seen_urls`` (set[str]).
    """

    def __init__(self, brief: Any) -> None:
        self.brief = brief
        self._seen_urls: set[str] = set()

    def _merge(self) -> None:
        from linkedin.orchestrator import Pipeline

        # Bind unbound method to this stub (we only need the helper, not
        # the rest of Pipeline's surface).
        Pipeline._merge_prior_search_exclusion(self)  # type: ignore[arg-type]


def test_merge_with_two_ruled_out_urls_adds_them_to_seen_urls(
    tmp_path: Path,
) -> None:
    brief = load_brief(
        _write_brief(
            tmp_path,
            _v2_brief_payload(
                ruled_out_urls=[
                    "https://linkedin.com/in/cand-a",
                    "https://linkedin.com/in/cand-b",
                ]
            ),
        )
    )
    stub = _StubOrchestrator(brief)
    stub._merge()
    assert "https://linkedin.com/in/cand-a" in stub._seen_urls
    assert "https://linkedin.com/in/cand-b" in stub._seen_urls


def test_merge_is_idempotent(tmp_path: Path) -> None:
    brief = load_brief(
        _write_brief(
            tmp_path,
            _v2_brief_payload(ruled_out_urls=["https://linkedin.com/in/x"]),
        )
    )
    stub = _StubOrchestrator(brief)
    stub._merge()
    once = set(stub._seen_urls)
    stub._merge()
    twice = set(stub._seen_urls)
    assert once == twice


def test_merge_no_op_when_brief_has_no_prior_search() -> None:
    """Classic briefs without the Slice 1 schema extension don't
    crash the merge."""

    class _NoPriorSearchBrief:
        pass

    stub = _StubOrchestrator(_NoPriorSearchBrief())
    stub._merge()
    assert stub._seen_urls == set()


def test_merge_no_op_when_ruled_out_urls_is_empty(tmp_path: Path) -> None:
    brief = load_brief(_write_brief(tmp_path, _v2_brief_payload(ruled_out_urls=[])))
    stub = _StubOrchestrator(brief)
    stub._merge()
    assert stub._seen_urls == set()


def test_merge_tolerates_malformed_ruled_out_urls(tmp_path: Path) -> None:
    """A brief that carries garbage in `ruled_out_urls` is loaded
    defensively; merge doesn't crash."""

    class _MalformedBrief:
        prior_search = PriorSearchContext(ruled_out_urls=[])

    # Force the field to a non-list at runtime so we exercise the
    # defensive branch that does NOT crash.
    object.__setattr__(_MalformedBrief.prior_search, "ruled_out_urls", "not a list")
    stub = _StubOrchestrator(_MalformedBrief())
    stub._merge()
    assert stub._seen_urls == set()


def test_merge_skips_non_string_url_entries(tmp_path: Path) -> None:
    """Defensive: list elements that aren't strings drop out."""

    class _BriefWithMixedTypes:
        prior_search = PriorSearchContext(
            ruled_out_urls=[
                "https://linkedin.com/in/valid",
                42,  # type: ignore[list-item]
                "",
                None,  # type: ignore[list-item]
                "https://linkedin.com/in/also-valid",
            ]
        )

    stub = _StubOrchestrator(_BriefWithMixedTypes())
    stub._merge()
    assert stub._seen_urls == {
        "https://linkedin.com/in/valid",
        "https://linkedin.com/in/also-valid",
    }


# ---------------------------------------------------------------------------
# End-to-end characterization
# ---------------------------------------------------------------------------


def test_e2e_exec_brief_loads_with_every_slice_1_through_9_field_intact(
    tmp_path: Path,
) -> None:
    """Slice 10 e2e characterization: the full executive-search brief
    shape (target_modules, confidentiality_class, prior_search,
    executive_calibration, dossier_spend_cap_usd, etc.) loads
    cleanly and every field downstream consumers depend on is
    accessible.

    This test guards against silent regressions in any of the
    schema additions across Slices 1-9.
    """

    payload = _v2_brief_payload(
        ruled_out_urls=[
            "https://linkedin.com/in/cand-a",
            "https://linkedin.com/in/cand-b",
        ],
        target_modules=["linkedin", "exec_search"],
    )
    payload["dossier_spend_cap_usd"] = 350.0
    payload["company_stage_signals"] = {"target_stage": "series_d"}
    payload["board_signals"] = {
        "relevant_board_companies": ["AcmeCorp"],
        "adjacency_rationale": "Two board overlaps.",
    }
    payload["executive_movement_window_days"] = 90

    brief = load_brief(_write_brief(tmp_path, payload))
    new_brief = brief._new_brief

    # Slice 1: confidentiality_class + prior_search + board_signals +
    # executive_movement_window_days + executive_calibration.
    assert brief.confidentiality_class == "blind"
    assert new_brief.confidentiality_class == "blind"
    assert brief.prior_search.ruled_out_urls == [
        "https://linkedin.com/in/cand-a",
        "https://linkedin.com/in/cand-b",
    ]
    assert new_brief.board_signals.relevant_board_companies == ["AcmeCorp"]
    assert new_brief.executive_movement_window_days == 90
    assert new_brief.executive_calibration is not None
    assert new_brief.executive_calibration.sector == "Healthcare"

    # Slice 2: target_modules + dossier_mode property.
    assert brief.target_modules == ["linkedin", "exec_search"]
    assert new_brief.target_modules == ["linkedin", "exec_search"]
    assert new_brief.dossier_mode is True

    # Slice 5: dossier_spend_cap_usd + company_stage_signals.
    assert new_brief.dossier_spend_cap_usd == pytest.approx(350.0)
    assert new_brief.company_stage_signals == {"target_stage": "series_d"}


def test_e2e_classic_linkedin_brief_unaffected_by_exec_search_extensions(
    tmp_path: Path,
) -> None:
    """Characterization regression: a classic brief without
    exec_search extensions hits all the dataclass defaults and
    behaves byte-identically to pre-Slice-1 loading.

    Slice 2's `dossier_mode` is False; Slice 10's prior-search
    merge is a no-op.
    """

    payload = _v2_brief_payload(
        ruled_out_urls=[],
        target_modules=["linkedin"],
    )
    # Drop the exec-specific knobs.
    del payload["confidentiality_class"]
    del payload["executive_calibration"]
    del payload["prior_search"]

    brief = load_brief(_write_brief(tmp_path, payload))
    new_brief = brief._new_brief

    assert new_brief.dossier_mode is False
    assert new_brief.confidentiality_class == "open"
    assert new_brief.executive_calibration is None
    assert new_brief.prior_search.ruled_out_urls == []
    # Slice 10 merge is a no-op for a classic brief.
    stub = _StubOrchestrator(brief)
    stub._merge()
    assert stub._seen_urls == set()
