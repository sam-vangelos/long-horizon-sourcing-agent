"""Designer Slice 2 — query formation from V2 brief.

Pins the contract for :func:`designer.strategy.form_designer_strategy`:

- Walks `capability_areas`, emits one Behance query per area as a baseline.
- Appends `behance_specialization_signals` as their own queries.
- Cross-products top spec signals × top tool signals up to the per-
  capability cap.
- Dedups by `(source, query_text.lower())`.
- Threads `discipline` from the dominant rubric calibration exemplar.
- Threads geography → ISO country code into `extra_filters` when the
  brief carries a recognizable geography string.
- Slice-2 returns Behance-only queries; CSE branch is a no-op.

Determinism matters: re-running the strategy on the same brief MUST
yield the same queries in the same order so work-unit IDs stay stable
across resumes.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from shared.schemas import ExecutionPlan

from designer.schemas import DesignerSearchQuery
from designer.strategy import (
    BEHANCE_SORT_PRIMARY,
    MAX_QUERIES_PER_CAPABILITY_AREA,
    form_designer_strategy,
    form_strategy_for_registry,
)


@pytest.fixture(autouse=True)
def _behance_only_default_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit Move #14 backward-compat: this test file pre-dates the
    CSE-primary contract and asserts Behance-specific query formation.
    Set BEHANCE_API_KEY (and clear CSE keys) so
    ``select_designer_sources()`` returns ``("behance",)`` — making
    every existing test's bare ``form_designer_strategy(brief)`` call
    behave the same way it did pre-Move-14. Tests that exercise the
    Move-14 default pass ``sources=...`` explicitly and override this.
    """

    monkeypatch.setenv("BEHANCE_API_KEY", "behance-test-key")
    monkeypatch.delenv("GOOGLE_CSE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CSE_ID", raising=False)


def _brief_with_one_capability_area() -> dict:
    return {
        "role_title": "Senior product designer",
        "capability_areas": [
            {
                "name": "Design systems",
                "description": "Builds and maintains design systems.",
                "behance_specialization_signals": [
                    "design systems",
                    "component library",
                ],
                "tool_stack_signals": ["Figma", "Storybook"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Owns the design system end-to-end.",
            "user_definition": "Consumes the system without authoring it.",
            "edge_case_guidance": "Borderline = consuming + extending.",
        },
    }


def _brief_with_calibration_exemplars(
    *exemplar_disciplines: str,
) -> dict:
    brief = _brief_with_one_capability_area()
    brief["design_rubric"] = {
        "calibration_exemplars": [
            {
                "portfolio_url": f"https://example.com/p{idx}",
                "discipline": discipline,
                "verdict": "yes",
                "per_principle_reasoning": {},
                "overall_reasoning": "fixture",
            }
            for idx, discipline in enumerate(exemplar_disciplines)
        ]
    }
    return brief


def test_strategy_emits_baseline_query_for_capability_area_name() -> None:
    queries = form_designer_strategy(_brief_with_one_capability_area())
    # Baseline query is the capability-area name itself.
    baseline = [q for q in queries if q.query_text == "Design systems"]
    assert len(baseline) == 1
    assert baseline[0].source == "behance"
    assert baseline[0].sort == BEHANCE_SORT_PRIMARY
    assert baseline[0].capability_area_name == "Design systems"


def test_strategy_emits_one_query_per_specialization_signal() -> None:
    """Each specialization signal becomes a distinct query — except
    when it duplicates the capability-area name under case-insensitive
    dedup (Behance search is case-insensitive, so the dedup at the
    strategy layer matches that behavior)."""

    queries = form_designer_strategy(_brief_with_one_capability_area())
    query_texts_lower = {q.query_text.lower() for q in queries}
    assert "design systems" in query_texts_lower  # baseline OR signal — dedupes
    assert "component library" in query_texts_lower


def test_strategy_emits_signal_x_tool_combinations() -> None:
    """Each top-2 specialization × top-2 tool produces a combined query,
    bounded by the per-capability cap. With 1 baseline + 2 signals +
    up to 4 combos, the cap of 6 trims the last combo."""

    queries = form_designer_strategy(_brief_with_one_capability_area())
    query_texts = {q.query_text for q in queries}
    # At least these three combos land within the cap.
    assert "design systems Figma" in query_texts
    assert "design systems Storybook" in query_texts
    assert "component library Figma" in query_texts
    # Total query count respects the per-capability cap.
    assert len(queries) <= MAX_QUERIES_PER_CAPABILITY_AREA


def test_strategy_caps_queries_per_capability_area() -> None:
    """A pathologically broad capability area must not emit more than
    `MAX_QUERIES_PER_CAPABILITY_AREA` queries."""

    brief = {
        "role_title": "Senior designer",
        "capability_areas": [
            {
                "name": "Broad area",
                "description": "Many signals.",
                "behance_specialization_signals": [f"signal_{i}" for i in range(8)],
                "tool_stack_signals": [f"tool_{i}" for i in range(8)],
            }
        ],
        "depth_distinction": {"builder_definition": "x", "user_definition": "y", "edge_case_guidance": "z"},
    }
    queries = form_designer_strategy(brief)
    queries_for_area = [q for q in queries if q.capability_area_name == "Broad area"]
    assert len(queries_for_area) <= MAX_QUERIES_PER_CAPABILITY_AREA


def test_strategy_dedups_duplicate_query_text() -> None:
    """If two capability areas surface the same specialization signal,
    only one query goes out — work-unit dedup at the strategy layer."""

    brief = {
        "role_title": "Designer",
        "capability_areas": [
            {
                "name": "Area A",
                "description": "first",
                "behance_specialization_signals": ["overlap signal"],
            },
            {
                "name": "Area B",
                "description": "second",
                "behance_specialization_signals": ["overlap signal"],
            },
        ],
        "depth_distinction": {"builder_definition": "x", "user_definition": "y", "edge_case_guidance": "z"},
    }
    queries = form_designer_strategy(brief)
    overlap_queries = [q for q in queries if q.query_text == "overlap signal"]
    assert len(overlap_queries) == 1


def test_strategy_threads_dominant_discipline_from_calibration_exemplars() -> None:
    brief = _brief_with_calibration_exemplars("product", "product", "product", "brand")
    queries = form_designer_strategy(brief)
    # Every query carries the dominant discipline (product).
    for query in queries:
        assert query.discipline == "product"


def test_strategy_handles_no_calibration_exemplars() -> None:
    queries = form_designer_strategy(_brief_with_one_capability_area())
    for query in queries:
        assert query.discipline == ""


def test_strategy_threads_geography_iso_code_into_extra_filters() -> None:
    brief = _brief_with_one_capability_area()
    brief["geography"] = "US"
    queries = form_designer_strategy(brief)
    for query in queries:
        assert query.extra_filters.get("country") == "US"


def test_strategy_translates_natural_language_geography() -> None:
    brief = _brief_with_one_capability_area()
    brief["geography"] = "United Kingdom"
    queries = form_designer_strategy(brief)
    for query in queries:
        assert query.extra_filters.get("country") == "GB"


def test_strategy_omits_country_filter_for_unknown_geography() -> None:
    brief = _brief_with_one_capability_area()
    brief["geography"] = "Outer Space"
    queries = form_designer_strategy(brief)
    for query in queries:
        assert "country" not in query.extra_filters


def test_strategy_is_deterministic_across_runs() -> None:
    """Same brief in → same queries (and same order) out. Required for
    work-unit ID stability across resume."""

    brief = _brief_with_one_capability_area()
    first = form_designer_strategy(brief)
    second = form_designer_strategy(brief)
    assert first == second


def test_strategy_returns_empty_for_brief_without_capability_areas() -> None:
    brief = {
        "role_title": "Designer",
        "depth_distinction": {"builder_definition": "x", "user_definition": "y", "edge_case_guidance": "z"},
    }
    assert form_designer_strategy(brief) == []


def test_strategy_handles_empty_signals_lists_gracefully() -> None:
    brief = {
        "role_title": "Designer",
        "capability_areas": [
            {
                "name": "Bare area",
                "description": "no signals",
                "behance_specialization_signals": [],
                "tool_stack_signals": [],
            }
        ],
        "depth_distinction": {"builder_definition": "x", "user_definition": "y", "edge_case_guidance": "z"},
    }
    queries = form_designer_strategy(brief)
    # Just the baseline capability-area-name query.
    assert len(queries) == 1
    assert queries[0].query_text == "Bare area"


# ---------------------------------------------------------------------------
# Multi-Agent Execution Plan Slice 1.6 — registry adapter
# ---------------------------------------------------------------------------


def test_form_strategy_for_registry_returns_execution_plan() -> None:
    """The registry adapter normalizes Designer's deterministic
    ``list[DesignerSearchQuery]`` output into the
    :class:`shared.schemas.ExecutionPlan` shape consumed by
    ``cloris.launchers.LauncherEntry.form_strategy_fn``."""

    brief = _brief_with_one_capability_area()
    plan = form_strategy_for_registry(brief)

    assert isinstance(plan, ExecutionPlan)
    # Designer's strategy is deterministic; the adapter does not
    # editorialize a rationale string. (Correction 3a.)
    assert plan.strategy_rationale == ""


def test_form_strategy_for_registry_preserves_native_query_shape() -> None:
    """Adapter output ``generated_strings`` must round-trip the native
    ``form_designer_strategy`` queries field-for-field; no information
    is lost in the wrapping."""

    brief = _brief_with_one_capability_area()
    native_queries = form_designer_strategy(brief)
    plan = form_strategy_for_registry(brief)

    assert plan.generated_strings == [asdict(q) for q in native_queries]


def test_form_strategy_for_registry_unwraps_compat_brief_via_new_brief() -> None:
    """Cross-module callers (chief-of-staff dispatch in Phase 2.5) hand
    in the compat :class:`shared.brief_loader.Brief`, which carries the
    V2 dict on ``_new_brief``. The adapter unwraps it transparently."""

    raw = _brief_with_one_capability_area()

    class _CompatBrief:
        _new_brief = raw

    direct = form_strategy_for_registry(raw)
    via_compat = form_strategy_for_registry(_CompatBrief())

    assert direct.generated_strings == via_compat.generated_strings


def test_form_strategy_for_registry_ignores_prior_run_data() -> None:
    """Designer's strategy is a pure function of brief content;
    ``prior_run_data`` is accepted for registry-signature uniformity
    but does not perturb output."""

    brief = _brief_with_one_capability_area()
    without = form_strategy_for_registry(brief)
    with_prior = form_strategy_for_registry(brief, prior_run_data={"saves": 7})

    assert without.generated_strings == with_prior.generated_strings


def test_strategy_emits_only_requested_source_set() -> None:
    """`form_designer_strategy(..., sources=("google_cse",))` emits
    only CSE queries (Slice 3 wires the CSE branch); Slice 10 will
    add Dribbble. Sources outside the requested set are filtered."""

    cse_only = form_designer_strategy(
        _brief_with_one_capability_area(),
        sources=("google_cse",),
    )
    assert cse_only and all(q.source == "google_cse" for q in cse_only)

    behance_only = form_designer_strategy(
        _brief_with_one_capability_area(),
        sources=("behance",),
    )
    assert behance_only and all(q.source == "behance" for q in behance_only)
