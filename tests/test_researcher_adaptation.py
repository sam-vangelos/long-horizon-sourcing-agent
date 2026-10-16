from types import SimpleNamespace

from researcher.strategy import (
    ResearcherAdaptiveReport,
    ResearcherQueryReport,
    adapt_after_research_batch,
)
from shared.adaptive import AdaptiveAction


def _brief() -> SimpleNamespace:
    return SimpleNamespace(
        role_title="Research Scientist",
        role_summary="Post-training and evaluation research.",
        _new_brief={"source_config": {"researcher": {"discipline": "ml_general"}}},
    )


def test_researcher_adaptation_broadens_sparse_scout_with_heuristic_fallback() -> None:
    report = ResearcherAdaptiveReport(
        batch_name="Batch (1 researcher query)",
        query_reports=[
            ResearcherQueryReport(
                query_id=1,
                name="Venue-heavy sparse lane",
                topic_concepts=["C1"],
                venue_filter=["NeurIPS"],
                min_year=2024,
                min_citations=20,
                candidates_discovered=0,
            )
        ],
    )
    remaining = [
        {
            "id": 2,
            "name": "ICML lane",
            "topic_concepts": ["C2"],
            "venue_filter": ["ICML"],
            "min_year": 2024,
            "min_citations": 20,
            "ror_country_filter": ["US"],
        }
    ]

    plan = adapt_after_research_batch(
        _brief(),
        report,
        remaining,
        llm_caller=lambda _system, _user: (_ for _ in ()).throw(RuntimeError("down")),
    )

    assert plan.decision is not None
    assert plan.decision.action is AdaptiveAction.BROADEN
    assert plan.new_queries
    assert plan.new_queries[0]["venue_filter"] == []
    assert plan.new_queries[0]["min_citations"] == 0
    assert plan.decision.source_payload["new_queries"][0]["adapted_reason"] == (
        "sparse_results_broadened_filters"
    )


def test_researcher_adaptation_parses_native_skip_and_new_queries() -> None:
    report = ResearcherAdaptiveReport(
        batch_name="Batch (1 researcher query)",
        query_reports=[
            ResearcherQueryReport(
                query_id=1,
                name="Noisy broad concept",
                topic_concepts=["C1"],
                candidates_discovered=12,
                facial_no_count=12,
            )
        ],
    )
    remaining = [
        {
            "id": 2,
            "name": "Queued broad concept",
            "topic_concepts": ["C1"],
            "venue_filter": [],
            "min_year": 2023,
            "min_citations": 0,
            "ror_country_filter": [],
        }
    ]

    plan = adapt_after_research_batch(
        _brief(),
        report,
        remaining,
        llm_caller=lambda _system, _user: {
            "new_researcher_queries": [
                {
                    "name": "Narrowed venue lane",
                    "topic_concepts": ["C1"],
                    "venue_filter": ["ACL"],
                    "min_year": 2023,
                    "min_citations": 5,
                    "ror_country_filter": ["US"],
                }
            ],
            "skip_query_ids": [2],
            "rationale": "Broad concept was noisy; narrow to ACL and skip duplicate broad lane.",
        },
    )

    assert plan.decision is not None
    assert plan.decision.action is AdaptiveAction.NARROW
    assert plan.skipped_query_ids == [2]
    assert plan.new_queries[0]["venue_filter"] == ["ACL"]
    assert plan.decision.skipped_work_units == ["2"]


def test_researcher_classifier_narrow_wins_when_both_sparse_and_noisy_fire() -> None:
    # When a batch surfaces both sparse AND noisy queries and the planner
    # proposes a skip plus a new query, NARROW should win over BROADEN —
    # the noise-cut is more urgent than the broaden.
    report = ResearcherAdaptiveReport(
        batch_name="Mixed batch",
        query_reports=[
            ResearcherQueryReport(
                query_id=1,
                name="Sparse venue lane",
                topic_concepts=["C1"],
                venue_filter=["NeurIPS"],
                candidates_discovered=0,
            ),
            ResearcherQueryReport(
                query_id=2,
                name="Noisy broad concept",
                topic_concepts=["C2"],
                candidates_discovered=12,
                facial_no_count=12,
            ),
        ],
    )
    remaining = [
        {
            "id": 3,
            "name": "Queued broad concept",
            "topic_concepts": ["C2"],
            "venue_filter": [],
            "min_year": 2023,
            "min_citations": 0,
            "ror_country_filter": [],
        }
    ]

    plan = adapt_after_research_batch(
        _brief(),
        report,
        remaining,
        llm_caller=lambda _system, _user: {
            "new_researcher_queries": [
                {
                    "name": "Narrowed venue lane",
                    "topic_concepts": ["C2"],
                    "venue_filter": ["ACL"],
                    "min_year": 2023,
                    "min_citations": 5,
                    "ror_country_filter": ["US"],
                }
            ],
            "skip_query_ids": [3],
            "rationale": "Noisy concept; narrow to ACL.",
        },
    )

    assert plan.decision is not None
    assert plan.decision.action is AdaptiveAction.NARROW
