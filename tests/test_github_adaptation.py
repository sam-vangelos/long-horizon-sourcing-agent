from github.schemas import GitHubBatchReport, GitHubSearchQuery
from github.strategy import build_github_adaptation_decision
from shared.adaptive import AdaptiveAction


def test_github_adaptation_decision_carries_source_native_queries() -> None:
    report = GitHubBatchReport(
        batch_name="Batch (2 queries)",
        queries_run=2,
        total_candidates_discovered=17,
        total_saves=2,
        total_rejects=3,
        total_insufficient=1,
        top_performing_queries=[
            {"query_id": 1, "name": "Repo maintainers", "saves": 2}
        ],
        zero_save_query_ids=[2],
        query_details=[
            {
                "query_id": 1,
                "name": "Repo maintainers",
                "query_string": "",
                "channel": "repo_mining",
                "saves": 2,
                "candidates": 9,
            },
            {
                "query_id": 2,
                "name": "Broad users",
                "query_string": "language:python",
                "channel": "user_search",
                "saves": 0,
                "candidates": 8,
            },
        ],
        channel_metrics=[
            {"channel": "repo_mining", "queries": 1, "candidates": 9, "saves": 2},
            {"channel": "user_search", "queries": 1, "candidates": 8, "saves": 0},
        ],
        signal_markers=[
            {
                "kind": "productive_query",
                "label": "queries with saves",
                "count": 1,
                "examples": ["Repo maintainers"],
            }
        ],
        noise_markers=[
            {
                "kind": "zero_save_query",
                "label": "queries with zero saves",
                "count": 1,
                "examples": ["Broad users"],
            }
        ],
    )
    new_queries = [
        GitHubSearchQuery(
            id=99,
            name="Adjacent maintainer repo",
            query="",
            channel="repo_mining",
            target_repo="owner/repo",
        )
    ]

    decision = build_github_adaptation_decision(
        batch_report=report,
        new_queries=new_queries,
        rationale="Repo mining is producing saves; add adjacent repos.",
        skipped_ids=[2],
    )

    assert decision.action is AdaptiveAction.EXPERIMENT
    assert decision.inserted_work_units == ["99"]
    assert decision.skipped_work_units == ["2"]
    assert decision.metrics.saves == 2
    assert decision.metrics.signal_markers[0].kind == "productive_query"
    assert decision.source_payload["new_queries"][0]["target_repo"] == "owner/repo"
    # repo_mining had the saves, so the dominant channel wins the lane.
    assert decision.lane == "repo_mining"
    assert "," not in decision.lane
    assert decision.work_unit_family == "repo_mining"


def test_github_classifier_returns_narrow_when_cutting_noise_and_adding_tighter_queries() -> None:
    # A batch that produced no saves AND the planner emitted both skips
    # AND tighter new queries should classify as NARROW — cut noise + add
    # precision. Locks in the behavior so a future classifier refactor
    # can't silently regress it.
    report = GitHubBatchReport(
        batch_name="Batch (1 query)",
        queries_run=1,
        total_candidates_discovered=8,
        total_saves=0,
        total_rejects=8,
        total_insufficient=0,
        zero_save_query_ids=[7],
        query_details=[
            {
                "query_id": 7,
                "name": "Noisy users",
                "query_string": "language:python",
                "channel": "user_search",
                "saves": 0,
                "candidates": 8,
            }
        ],
        channel_metrics=[
            {"channel": "user_search", "queries": 1, "candidates": 8, "saves": 0},
        ],
    )
    new_queries = [
        GitHubSearchQuery(
            id=42,
            name="Tighter maintainer query",
            query="language:python topic:llm",
            channel="user_search",
        )
    ]

    decision = build_github_adaptation_decision(
        batch_report=report,
        new_queries=new_queries,
        rationale="Broad user search was noisy; replace with topic-scoped query.",
        skipped_ids=[7],
    )

    assert decision.action is AdaptiveAction.NARROW
    assert decision.skipped_work_units == ["7"]
    assert decision.inserted_work_units == ["42"]
    assert decision.lane == "user_search"


def test_github_classifier_skip_only_does_not_drift_into_narrow() -> None:
    # A skip-only batch (no new queries) is a SKIP, not a NARROW —
    # narrowing implies adding tighter queries. Cross-source classifiers
    # share this convention.
    report = GitHubBatchReport(
        batch_name="Batch (1 query)",
        queries_run=1,
        total_candidates_discovered=8,
        total_saves=0,
        total_rejects=8,
        total_insufficient=0,
        zero_save_query_ids=[7],
        query_details=[
            {
                "query_id": 7,
                "name": "Noisy users",
                "query_string": "language:python",
                "channel": "user_search",
                "saves": 0,
                "candidates": 8,
            }
        ],
        channel_metrics=[
            {"channel": "user_search", "queries": 1, "candidates": 8, "saves": 0},
        ],
    )

    decision = build_github_adaptation_decision(
        batch_report=report,
        new_queries=[],
        rationale="Cut the noisy lane.",
        skipped_ids=[7],
    )

    assert decision.action is AdaptiveAction.SKIP
