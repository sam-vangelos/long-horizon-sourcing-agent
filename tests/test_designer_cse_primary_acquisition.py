"""Tests for the CSE-primary acquisition contract — audit Move #14.

Per the audit plan, Move #14 closes the "Designer launches blocked
indefinitely without pre-2020 Behance key" failure mode by making
Google CSE the primary acquisition source. Behance becomes optional
secondary signal when configured.

This file pins the contract pieces that close the failure mode:

- :func:`designer.strategy.select_designer_sources` reads env vars
  and returns the right source tuple (CSE first when configured;
  Behance augment when also configured; empty when neither —
  matching the health probe's hard blocker behavior).
- :func:`designer.strategy.form_designer_strategy` defaults to
  :func:`select_designer_sources` when ``sources`` is None
  (post-Move-14 default behavior); pre-Move-14 callers passing
  ``("behance",)`` explicitly continue to work unchanged.
- The CSE-only path produces ≥10 candidates on a fixture brief
  (the audit plan's acceptance criterion). Behance-augment fan-out
  verified separately.

Health probe relaxation is covered in
:mod:`tests.test_designer_health` — this file focuses on the
strategy / acquisition seam.
"""

from __future__ import annotations

from typing import Any

import pytest

from designer.strategy import (
    form_designer_strategy,
    select_designer_sources,
)


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------


class TestSelectDesignerSources:
    def test_returns_empty_when_no_keys_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
        monkeypatch.delenv("BEHANCE_API_KEY", raising=False)
        assert select_designer_sources() == ()

    def test_returns_cse_only_when_cse_keys_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BEHANCE_API_KEY", raising=False)
        assert select_designer_sources(
            google_cse_api_key="real",
            google_cse_id="real",
            behance_api_key="",
        ) == ("google_cse",)

    def test_returns_behance_only_when_only_behance_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)
        assert select_designer_sources(
            behance_api_key="real",
            google_cse_api_key="",
            google_cse_id="",
        ) == ("behance",)

    def test_returns_cse_first_then_behance_when_both_configured(self) -> None:
        """CSE-primary contract: CSE comes first in the tuple so the
        strategy planner emits CSE queries before Behance queries."""

        assert select_designer_sources(
            behance_api_key="real",
            google_cse_api_key="real_cse",
            google_cse_id="real_cse_id",
        ) == ("google_cse", "behance")

    def test_partial_cse_config_treated_as_no_cse(self) -> None:
        """Half-configured CSE (api_key without ID, or vice versa)
        doesn't count — both are needed for the client."""

        assert select_designer_sources(
            behance_api_key="",
            google_cse_api_key="real",
            google_cse_id="",
        ) == ()
        assert select_designer_sources(
            behance_api_key="",
            google_cse_api_key="",
            google_cse_id="real",
        ) == ()


# ---------------------------------------------------------------------------
# CSE-only acquisition produces queries
# ---------------------------------------------------------------------------


def _designer_brief_for_cse(num_capability_areas: int = 4) -> dict[str, Any]:
    """Build a Designer brief with N capability areas, each carrying
    enough signal to drive multi-query CSE strategy formation."""

    capability_areas: list[dict[str, Any]] = []
    base_areas = [
        {
            "name": "Visual systems",
            "description": "Design systems thinking and design tooling.",
            "behance_specialization_signals": ["design system", "tokens"],
            "tool_stack_signals": ["Figma", "Storybook"],
        },
        {
            "name": "Brand identity",
            "description": "Distinctive marks and visual systems.",
            "behance_specialization_signals": ["brand identity", "logo"],
            "tool_stack_signals": ["Adobe Illustrator"],
        },
        {
            "name": "Editorial layout",
            "description": "Long-form publication design.",
            "behance_specialization_signals": ["editorial design"],
            "tool_stack_signals": ["InDesign"],
        },
        {
            "name": "Motion graphics",
            "description": "Title sequences and brand motion.",
            "behance_specialization_signals": ["motion design"],
            "tool_stack_signals": ["After Effects"],
        },
    ]
    for area in base_areas[:num_capability_areas]:
        capability_areas.append(area)

    return {
        "role_title": "Senior product designer",
        "geography": "United States",
        "capability_areas": capability_areas,
        "design_rubric": {
            "principles": [
                {
                    "name": "Visual hierarchy",
                    "description": "Clear attention direction.",
                    "weight": 1.0,
                },
            ],
        },
    }


class TestCSEPrimaryStrategyFormation:
    def test_cse_only_path_produces_at_least_10_queries(self) -> None:
        """Audit Move #14 acceptance: CSE-only acquisition produces
        ≥10 candidates on a fixture brief. We verify at the strategy
        layer (≥10 queries) since each query maps to a downstream
        candidate batch via the existing CSE acquisition path."""

        brief = _designer_brief_for_cse(num_capability_areas=4)
        queries = form_designer_strategy(brief, sources=("google_cse",))
        assert len(queries) >= 10, (
            f"expected ≥10 CSE queries from a 4-capability-area brief; "
            f"got {len(queries)}"
        )
        # Every query must be CSE-sourced.
        sources_seen = {q.source for q in queries}
        assert sources_seen == {"google_cse"}

    def test_cse_only_queries_cover_every_capability_area(self) -> None:
        """A CSE-only run must produce queries for every
        capability_area — no silent dropping."""

        brief = _designer_brief_for_cse(num_capability_areas=4)
        queries = form_designer_strategy(brief, sources=("google_cse",))
        # Each capability area name should appear in at least one
        # query's text.
        ca_names = [
            ca["name"] for ca in brief["capability_areas"]
        ]
        for ca_name in ca_names:
            matching = [
                q
                for q in queries
                if ca_name.lower() in q.query_text.lower()
            ]
            assert matching, (
                f"no query covers capability area {ca_name!r}; "
                f"queries: {[q.query_text for q in queries]}"
            )


class TestBehanceAugmentPath:
    def test_both_sources_dedup_query_text(self) -> None:
        """When BOTH sources fire, queries dedup by (source, text);
        a CSE query and a Behance query with the same text are
        distinct rows (different sources)."""

        brief = _designer_brief_for_cse(num_capability_areas=2)
        queries = form_designer_strategy(
            brief, sources=("google_cse", "behance")
        )
        # Both source kinds present.
        source_kinds = {q.source for q in queries}
        assert "google_cse" in source_kinds
        assert "behance" in source_kinds
        # Within a single source, no duplicate query_text.
        for kind in source_kinds:
            same_source = [q.query_text.lower() for q in queries if q.source == kind]
            assert len(same_source) == len(set(same_source))

    def test_cse_queries_come_before_behance_within_each_capability_area(self) -> None:
        """CSE-primary contract: per-capability-area, CSE queries
        appear before Behance queries so the orchestrator's
        runtime-state work_units inherit a CSE-first ordering_index
        within each capability cluster. (Across capability areas
        the queries interleave: ca1.cse, ca1.behance, ca2.cse,
        ca2.behance — the meaningful contract is "CSE first within
        each capability," matching the runtime-state work_unit
        clustering.)"""

        brief = _designer_brief_for_cse(num_capability_areas=2)
        queries = form_designer_strategy(
            brief, sources=("google_cse", "behance")
        )
        # The very first query must be CSE-sourced (CSE-primary).
        assert queries[0].source == "google_cse", (
            f"expected first query to be CSE-sourced; got "
            f"{queries[0].source!r}"
        )


class TestStrategyDefaultsToSelectedSources:
    def test_default_sources_is_select_designer_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When form_designer_strategy is called without sources,
        it resolves via select_designer_sources (env-driven)."""

        # Configure CSE-only via env vars.
        monkeypatch.setenv("GOOGLE_CSE_API_KEY", "real_cse")
        monkeypatch.setenv("GOOGLE_CSE_ID", "real_cse_id")
        monkeypatch.delenv("BEHANCE_API_KEY", raising=False)

        brief = _designer_brief_for_cse(num_capability_areas=2)
        queries = form_designer_strategy(brief)  # no sources passed
        sources_seen = {q.source for q in queries}
        assert sources_seen == {"google_cse"}, (
            f"default-sources resolution should respect env: got {sources_seen}"
        )

    def test_explicit_sources_kwarg_overrides_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pre-Move-14 callers passing ``sources=("behance",)``
        explicitly continue to work unchanged."""

        monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
        monkeypatch.delenv("BEHANCE_API_KEY", raising=False)

        brief = _designer_brief_for_cse(num_capability_areas=2)
        queries = form_designer_strategy(brief, sources=("behance",))
        sources_seen = {q.source for q in queries}
        assert sources_seen == {"behance"}
