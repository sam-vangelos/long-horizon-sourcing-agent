"""Researcher module Slice 3 — strategy formation coverage.

`form_strategy(brief, prior_data) -> ExecutionPlan` produces queries
shaped per Researcher Module Spec Opinion 2 (NOT Boolean strings —
OpenAlex API parameters). Tests verify:

- Prompt assembly pulls in capability areas + source_config.researcher.
- LLM output is normalized: missing keys default; non-dict entries
  dropped; venue_filter falls back to conference_allowlist seed.
- The injected ``llm_caller`` is called with both the system + user
  prompts (no real Opus call from tests).
- The query schema matches :data:`RESEARCHER_QUERY_SCHEMA_KEYS`.
"""

from __future__ import annotations

from types import SimpleNamespace

from unittest.mock import patch

from shared.schemas import ExecutionPlan

from researcher.strategy import (
    RESEARCHER_QUERY_SCHEMA_KEYS,
    form_strategy,
    form_strategy_for_registry,
)


def _capability(name: str, description: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description=description)


def _stub_brief(
    *,
    role_title: str = "Frontier-lab Researcher",
    role_summary: str = "Original research on RLHF + agent infra.",
    capability_areas: list | None = None,
    source_config_researcher: dict | None = None,
) -> SimpleNamespace:
    """Build a minimal Brief-shape for testing strategy formation."""

    new_brief: dict = {}
    if source_config_researcher is not None:
        new_brief["source_config"] = {"researcher": source_config_researcher}
    return SimpleNamespace(
        role_title=role_title,
        role_summary=role_summary,
        capability_areas=capability_areas
        or [
            _capability(
                "Post-training research",
                "Publishes original work on RLHF / DPO / SFT.",
            ),
            _capability(
                "Agent infrastructure",
                "Designs reasoning + tool-use systems.",
            ),
        ],
        _new_brief=new_brief,
    )


def _stub_llm_response() -> dict:
    return {
        "strategy_rationale": "Cover post-training and inference axes via NeurIPS + ICML.",
        "generated_strings": [
            {
                "id": 1,
                "name": "RLHF · NeurIPS",
                "topic_concepts": ["C2778407487"],
                "venue_filter": ["NeurIPS"],
                "min_year": 2023,
                "min_citations": 20,
                "ror_country_filter": ["US", "GB"],
            },
            {
                "id": 2,
                "name": "Agent infra · ICML",
                "topic_concepts": ["C41008148", "C99498"],
                "venue_filter": ["ICML"],
                "min_year": 2023,
                "min_citations": 10,
                "ror_country_filter": [],
            },
            {
                "topic_concepts": ["C9000"],  # Missing id, name, venue_filter
            },
        ],
        "coverage_gaps": [],
        "architecture": "concept_first",
        "architecture_rationale": "Concept-first because brief names topics, not venues.",
    }


# ---------------------------------------------------------------------------
# Plan construction
# ---------------------------------------------------------------------------


def test_form_strategy_calls_llm_with_system_and_user_prompts() -> None:
    captured: dict = {}

    def llm_caller(system: str, user: str) -> dict:
        captured["system"] = system
        captured["user"] = user
        return _stub_llm_response()

    brief = _stub_brief(
        source_config_researcher={
            "research_topics": ["RLHF", "agent infrastructure"],
            "conference_allowlist": ["NeurIPS", "ICML", "ICLR"],
            "discipline": "nlp",
        }
    )
    plan = form_strategy(brief, llm_caller=llm_caller)

    assert "system" in captured and "user" in captured
    assert "RLHF, agent infrastructure" in captured["system"]
    assert "NeurIPS, ICML, ICLR" in captured["system"]
    assert "nlp" in captured["system"]
    # Capability areas appear by name in the system prompt.
    assert "Post-training research" in captured["system"]
    assert "Agent infrastructure" in captured["system"]
    # Output schema keys appear in the user prompt's contract.
    for key in RESEARCHER_QUERY_SCHEMA_KEYS:
        assert key in captured["user"]
    assert plan.strategy_rationale.startswith("Cover post-training")


def test_form_strategy_normalizes_query_dicts() -> None:
    brief = _stub_brief(
        source_config_researcher={
            "conference_allowlist": ["NeurIPS", "ICML"],
        }
    )
    plan = form_strategy(brief, llm_caller=lambda _s, _u: _stub_llm_response())

    queries = plan.generated_strings
    assert len(queries) == 3

    first = queries[0]
    assert first["id"] == 1
    assert first["name"] == "RLHF · NeurIPS"
    assert first["boolean"] == ""  # Spec Opinion 2: no Boolean for researcher.
    assert first["topic_concepts"] == ["C2778407487"]
    assert first["venue_filter"] == ["NeurIPS"]
    assert first["min_year"] == 2023
    assert first["min_citations"] == 20
    assert first["ror_country_filter"] == ["US", "GB"]

    third = queries[2]
    # Missing id auto-assigned from index.
    assert third["id"] == 3
    # Missing venue_filter falls back to conference_allowlist seed.
    assert third["venue_filter"] == ["NeurIPS", "ICML"]
    # Missing min_year / min_citations default to 0.
    assert third["min_year"] == 0
    assert third["min_citations"] == 0
    # Auto-named when LLM didn't supply.
    assert third["name"]


def test_form_strategy_handles_missing_generated_strings() -> None:
    """Audit Move #4 R7: an LLM plan that lacks generated_strings now
    triggers the heuristic fallback (rather than passing through as an
    empty plan that produces zero candidates downstream). The
    deterministic heuristic plan reads research_topics +
    conference_allowlist from source_config."""

    brief = _stub_brief(
        source_config_researcher={
            "research_topics": ["C1"],
            "conference_allowlist": ["NeurIPS"],
        }
    )
    plan = form_strategy(
        brief,
        llm_caller=lambda _s, _u: {"strategy_rationale": "no plan"},
    )
    assert "heuristic" in plan.strategy_rationale.lower()
    assert len(plan.generated_strings) == 1
    assert plan.generated_strings[0]["topic_concepts"] == ["C1"]
    assert plan.generated_strings[0]["venue_filter"] == ["NeurIPS"]


def test_form_strategy_drops_non_dict_query_entries() -> None:
    brief = _stub_brief()
    plan = form_strategy(
        brief,
        llm_caller=lambda _s, _u: {
            "generated_strings": [
                {"topic_concepts": ["C1"]},
                "not a dict",
                None,
                {"topic_concepts": ["C2"]},
            ]
        },
    )
    assert len(plan.generated_strings) == 2
    assert plan.generated_strings[0]["topic_concepts"] == ["C1"]
    assert plan.generated_strings[1]["topic_concepts"] == ["C2"]


def test_form_strategy_with_no_source_config_uses_empty_seed() -> None:
    """A brief without source_config.researcher should still produce a
    plan; the venue_filter just won't have a fallback seed.
    """

    brief = _stub_brief(source_config_researcher=None)
    plan = form_strategy(
        brief,
        llm_caller=lambda _s, _u: {
            "generated_strings": [{"topic_concepts": ["C1"]}]
        },
    )
    assert plan.generated_strings[0]["venue_filter"] == []


def test_form_strategy_passes_prior_run_data_into_user_prompt() -> None:
    captured_user: dict = {}

    def llm_caller(system: str, user: str) -> dict:
        captured_user["text"] = user
        return _stub_llm_response()

    brief = _stub_brief()
    form_strategy(
        brief,
        prior_data={
            "queries_explored": 7,
            "high_yield_venues": ["NeurIPS"],
        },
        llm_caller=llm_caller,
    )
    assert "queries explored: 7" in captured_user["text"]
    assert "high-yield venues: ['NeurIPS']" in captured_user["text"]


def test_form_strategy_prior_data_none_emits_fresh_marker() -> None:
    captured_user: dict = {}

    def llm_caller(system: str, user: str) -> dict:
        captured_user["text"] = user
        return _stub_llm_response()

    form_strategy(_stub_brief(), llm_caller=llm_caller)
    assert "fresh strategy" in captured_user["text"]


# ---------------------------------------------------------------------------
# Multi-Agent Execution Plan Slice 1.6 — registry adapter
# ---------------------------------------------------------------------------


def test_form_strategy_for_registry_matches_native_call_shape() -> None:
    """The registry adapter wraps :func:`form_strategy` with the uniform
    ``(brief, prior_run_data) -> ExecutionPlan`` signature; the
    underlying plan must be equivalent to the native call. The adapter
    routes through the production-default ``opus_llm_cached`` path
    (Slice A.2 cache-gap remediation), so the test patches that for
    both calls to assert plan parity."""

    brief = _stub_brief()
    response = _stub_llm_response()

    with patch(
        "shared.llm_clients.opus_llm_cached",
        return_value=response,
    ) as mock_opus:
        adapter_plan = form_strategy_for_registry(brief)
        native_plan = form_strategy(brief, prior_data=None)

    assert isinstance(adapter_plan, ExecutionPlan)
    assert adapter_plan.strategy_rationale == native_plan.strategy_rationale
    assert adapter_plan.generated_strings == native_plan.generated_strings
    assert adapter_plan.architecture == native_plan.architecture
    # Both calls hit the same LLM seam — no extra Opus burn from the
    # adapter wrapping.
    assert mock_opus.call_count == 2


def test_form_strategy_for_registry_threads_prior_run_data() -> None:
    """The registry contract names the kwarg ``prior_run_data``; the
    adapter must forward it under the native ``prior_data`` kwarg
    name without losing the payload."""

    captured_user: dict = {}

    def fake_opus(system: str, user: str, **_kwargs) -> dict:
        captured_user["text"] = user
        return _stub_llm_response()

    # Slice A.2: researcher strategy now defaults to opus_llm_cached.
    with patch("shared.llm_clients.opus_llm_cached", side_effect=fake_opus):
        form_strategy_for_registry(
            _stub_brief(),
            prior_run_data={
                "queries_explored": 9,
                "high_yield_venues": ["ICML"],
            },
        )

    assert "queries explored: 9" in captured_user["text"]
    assert "high-yield venues: ['ICML']" in captured_user["text"]


def test_form_strategy_for_registry_does_not_merge_linkedin_lane_templates() -> None:
    with patch(
        "shared.llm_clients.opus_llm_cached",
        return_value=_stub_llm_response(),
    ):
        plan = form_strategy_for_registry(_stub_brief())

    assert plan.role_strategy_profile
    assert plan.sourcing_lanes == []
    assert plan.search_hypotheses == []
    assert plan.search_slices == []


# ---------------------------------------------------------------------------
# Audit Move #4 R7 — schema validation + heuristic fallback
# ---------------------------------------------------------------------------


from researcher.strategy import _heuristic_plan, _validate_plan_schema


class TestValidatePlanSchema:
    def test_returns_none_for_valid_plan(self) -> None:
        assert _validate_plan_schema(_stub_llm_response()) is None

    def test_returns_diagnostic_when_result_is_not_dict(self) -> None:
        assert "result_not_dict" in (_validate_plan_schema("nope") or "")
        assert "result_not_dict" in (_validate_plan_schema([]) or "")
        assert "result_not_dict" in (_validate_plan_schema(None) or "")

    def test_returns_diagnostic_when_generated_strings_missing(self) -> None:
        assert (
            _validate_plan_schema({"strategy_rationale": "yes"})
            == "generated_strings_empty_or_missing"
        )

    def test_returns_diagnostic_when_generated_strings_empty(self) -> None:
        assert (
            _validate_plan_schema({"generated_strings": []})
            == "generated_strings_empty_or_missing"
        )

    def test_returns_diagnostic_when_every_query_lacks_filters(self) -> None:
        assert _validate_plan_schema(
            {
                "generated_strings": [
                    {"id": 1, "name": "no-filters"},
                    {"id": 2, "topic_concepts": [], "venue_filter": []},
                ]
            }
        ) == "every_query_missing_filters"

    def test_passes_when_at_least_one_query_has_topic_concepts(self) -> None:
        assert _validate_plan_schema(
            {
                "generated_strings": [
                    {"id": 1, "topic_concepts": ["C1"]},
                    {"id": 2, "topic_concepts": []},  # invalid alone
                ]
            }
        ) is None

    def test_passes_when_query_has_only_venue_filter(self) -> None:
        assert _validate_plan_schema(
            {
                "generated_strings": [{"id": 1, "venue_filter": ["NeurIPS"]}]
            }
        ) is None


class TestHeuristicFallback:
    def test_form_strategy_falls_back_when_llm_raises(self) -> None:
        def _raises_llm(_system: str, _user: str) -> dict:
            raise RuntimeError("llm outage")

        plan = form_strategy(
            _stub_brief(
                source_config_researcher={
                    "research_topics": ["C1", "C2"],
                    "conference_allowlist": ["NeurIPS"],
                    "discipline": "ml_general",
                }
            ),
            llm_caller=_raises_llm,
        )
        # Heuristic plan: one query, populated from source_config.
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.generated_strings) == 1
        query = plan.generated_strings[0]
        assert query["topic_concepts"] == ["C1", "C2"]
        assert query["venue_filter"] == ["NeurIPS"]

    def test_form_strategy_falls_back_when_plan_lacks_generated_strings(self) -> None:
        def _bad_plan_llm(_system: str, _user: str) -> dict:
            return {"strategy_rationale": "no queries here"}

        plan = form_strategy(
            _stub_brief(
                source_config_researcher={"research_topics": ["C1"]}
            ),
            llm_caller=_bad_plan_llm,
        )
        assert len(plan.generated_strings) == 1
        assert plan.generated_strings[0]["topic_concepts"] == ["C1"]

    def test_form_strategy_falls_back_when_every_query_lacks_filters(
        self,
    ) -> None:
        def _filterless_llm(_system: str, _user: str) -> dict:
            return {
                "generated_strings": [
                    {"id": 1, "name": "vague"},
                    {"id": 2, "name": "also vague"},
                ]
            }

        plan = form_strategy(
            _stub_brief(
                source_config_researcher={"research_topics": ["C1"]}
            ),
            llm_caller=_filterless_llm,
        )
        assert len(plan.generated_strings) == 1

    def test_heuristic_plan_uses_venue_only_when_topics_absent(self) -> None:
        plan = _heuristic_plan(
            _stub_brief(
                source_config_researcher={
                    "conference_allowlist": ["NeurIPS", "ICML"]
                }
            ),
            source_config={"conference_allowlist": ["NeurIPS", "ICML"]},
        )
        assert len(plan.generated_strings) == 1
        query = plan.generated_strings[0]
        assert query["topic_concepts"] == []
        assert query["venue_filter"] == ["NeurIPS", "ICML"]

    def test_heuristic_plan_carries_strategy_rationale(self) -> None:
        plan = _heuristic_plan(
            _stub_brief(
                source_config_researcher={"research_topics": ["C1"]}
            ),
            source_config={"research_topics": ["C1"]},
        )
        assert plan.strategy_rationale
        assert "heuristic" in plan.strategy_rationale.lower()

    def test_form_strategy_passes_through_when_llm_plan_is_valid(self) -> None:
        """Pre-R7 happy path stays unchanged: a valid LLM plan
        skips the fallback entirely."""

        plan = form_strategy(
            _stub_brief(),
            llm_caller=lambda _s, _u: _stub_llm_response(),
        )
        # _stub_llm_response declares 3 queries — all valid (have
        # topic_concepts AND venue_filter); fallback doesn't fire.
        assert len(plan.generated_strings) == 3
        # Carries the LLM's strategy_rationale, not the heuristic's.
        assert "post-training" in plan.strategy_rationale.lower()
