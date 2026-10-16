from github.reconciliation_input import build_identity_resolution_experiment_cohort
from linkedin.identity_resolution_experiment import (
    BLOCKER_CAPTCHA,
    BLOCKER_EMPTY_RESULTS,
    BLOCKER_LOGIN_WALL,
    StrategyHealthTracker,
    _parse_bing_query,
    build_strategy_order,
    choose_winning_strategy,
    summarize_experiment_rows,
)
from shared.identity_experiment_schemas import (
    IdentityResolutionExperimentLead,
    IdentityResolutionGoldLabel,
    StrategyExecutionRecord,
    SurfacedProfileCandidate,
)


def test_parse_bing_query_matches_exact_and_loose_strategy_shapes():
    lead = IdentityResolutionExperimentLead(
        github_username="ada",
        candidate_name="Ada-Lovelace",
        lookup_name="Ada Lovelace",
        location="NYC",
    )

    exact_query, exact_city = _parse_bing_query(lead, exact_city=True)
    loose_query, loose_city = _parse_bing_query(lead, exact_city=False)

    assert exact_query == 'site:linkedin.com/in "Ada Lovelace" "New York City Metropolitan Area"'
    assert loose_query == "site:linkedin.com/in Ada Lovelace New York City Metropolitan Area"
    assert exact_city == loose_city == "New York City Metropolitan Area"


def test_build_strategy_order_is_seeded_but_stable():
    lead = IdentityResolutionExperimentLead(
        github_username="ada",
        candidate_name="Ada Lovelace",
    )
    names = ["web_exact_city", "web_loose_city", "linkedin_people_city", "recruiter_name_city"]

    first = build_strategy_order(lead, names, seed=17)
    second = build_strategy_order(lead, names, seed=17)
    third = build_strategy_order(lead, names, seed=18)

    assert first == second
    assert sorted(first) == sorted(names)
    assert third != first


def test_strategy_health_tracker_aborts_on_consecutive_hard_blockers():
    tracker = StrategyHealthTracker()

    tracker.record(BLOCKER_LOGIN_WALL)
    assert tracker.should_abort() is False
    tracker.record(BLOCKER_LOGIN_WALL)
    assert tracker.should_abort() is True
    assert "two consecutive" in tracker.aborted_reason


def test_strategy_health_tracker_uses_rate_guard_after_minimum_attempts():
    tracker = StrategyHealthTracker()

    tracker.record(BLOCKER_EMPTY_RESULTS)
    tracker.record("")
    tracker.record("")
    tracker.record("")
    assert tracker.should_abort() is False
    tracker.record(BLOCKER_CAPTCHA)
    assert tracker.should_abort() is True
    assert "fatal blocker rate exceeded" in tracker.aborted_reason


def test_strategy_health_tracker_does_not_abort_on_consecutive_ui_failures():
    tracker = StrategyHealthTracker()

    tracker.record("ui_failure")
    assert tracker.should_abort() is False
    tracker.record("ui_failure")
    assert tracker.should_abort() is False


def test_summarize_experiment_rows_applies_thresholds_and_picks_winner():
    rows = [
        StrategyExecutionRecord(
            strategy_name="web_exact_city",
            github_username=f"lead-{idx}",
            candidate_name=f"Ada {idx}",
            cohort_kind="primary",
            cohort_bucket="easy_exact_name",
            top1_correct=True,
            top3_contains_correct=True,
            duration_seconds=20,
            interaction_count=2,
        )
        for idx in range(8)
    ]
    rows.extend(
        [
            StrategyExecutionRecord(
                strategy_name="web_exact_city",
                github_username="lead-8",
                candidate_name="Ada 8",
                cohort_kind="primary",
                cohort_bucket="common_name_ambiguous",
                top1_correct=False,
                top3_contains_correct=True,
                duration_seconds=25,
                interaction_count=3,
                manual_review_required=True,
            ),
            StrategyExecutionRecord(
                strategy_name="web_exact_city",
                github_username="lead-9",
                candidate_name="Ada 9",
                cohort_kind="primary",
                cohort_bucket="name_variant",
                top1_correct=False,
                top3_contains_correct=True,
                duration_seconds=25,
                interaction_count=3,
                manual_review_required=True,
            ),
        ]
    )
    rows.extend(
        [
            StrategyExecutionRecord(
                strategy_name="web_loose_city",
                github_username=f"lead-{idx}",
                candidate_name=f"Ada {idx}",
                cohort_kind="primary",
                cohort_bucket="easy_exact_name",
                top1_correct=(idx < 5),
                top3_contains_correct=(idx < 7),
                wrong_person_top1=(idx == 7),
                duration_seconds=35,
                interaction_count=4,
                manual_review_required=(idx >= 7),
                no_candidate=(idx == 9),
            )
            for idx in range(10)
        ]
    )

    summary = summarize_experiment_rows(rows)

    assert summary["strategies"]["web_exact_city"]["default_eligible"] is True
    assert summary["strategies"]["web_loose_city"]["viable"] is False
    assert summary["decision"]["winner"] == "web_exact_city"


def test_choose_winning_strategy_prefers_phase_two_when_two_are_close():
    decision = choose_winning_strategy(
        {
            "web_exact_city": {
                "default_eligible": True,
                "viable": True,
                "aggregate_metrics": {
                    "top1_correct_rate": 0.70,
                    "wrong_person_top1_rate": 0.02,
                },
            },
            "linkedin_people_city": {
                "default_eligible": True,
                "viable": True,
                "aggregate_metrics": {
                    "top1_correct_rate": 0.67,
                    "wrong_person_top1_rate": 0.01,
                },
            },
        }
    )

    assert decision["winner"] == ""
    assert decision["decision"] == "run_phase_2_hybrid"


def test_scoreable_rows_for_exact_gold_support_public_url_and_manual_review():
    gold = IdentityResolutionGoldLabel(
        github_username="ada",
        candidate_name="Ada Lovelace",
        cohort_kind="primary",
        cohort_bucket="easy_exact_name",
        gold_outcome="exact_match",
        gold_public_linkedin_url="https://www.linkedin.com/in/ada-lovelace/",
        gold_display_name="Ada Lovelace",
        gold_company="Anthropic",
        gold_location="New York City Metropolitan Area",
        gold_title="Research Engineer",
    )
    row = StrategyExecutionRecord(
        strategy_name="web_exact_city",
        github_username="ada",
        candidate_name="Ada Lovelace",
        cohort_kind="primary",
        cohort_bucket="easy_exact_name",
        surfaced_candidates=[
            SurfacedProfileCandidate(
                profile_url="https://www.linkedin.com/in/ada-lovelace/",
                public_profile_url="https://www.linkedin.com/in/ada-lovelace/",
                display_name="Ada Lovelace",
                company="Anthropic",
                location="New York, New York, United States",
                headline="Research Engineer",
                rank=1,
            )
        ],
    )

    from linkedin.identity_resolution_experiment import score_execution_record

    scored = score_execution_record(row, gold)

    assert scored.top1_correct is True
    assert scored.top3_contains_correct is True
    assert scored.manual_review_required is False


def test_summarize_experiment_rows_tracks_aborted_rows_separately():
    rows = [
        StrategyExecutionRecord(
            strategy_name="web_exact_city",
            github_username="ada",
            candidate_name="Ada Lovelace",
            cohort_kind="primary",
            cohort_bucket="easy_exact_name",
            top1_correct=True,
            top3_contains_correct=True,
            duration_seconds=12.0,
            interaction_count=2,
        ),
        StrategyExecutionRecord(
            strategy_name="web_exact_city",
            github_username="grace",
            candidate_name="Grace Hopper",
            cohort_kind="primary",
            cohort_bucket="easy_exact_name",
            aborted=True,
            abort_reason="blocker rate exceeded 10%",
            blocker_state="aborted",
        ),
    ]

    summary = summarize_experiment_rows(rows)

    assert summary["strategies"]["web_exact_city"]["attempted_leads"] == 1
    assert summary["strategies"]["web_exact_city"]["aborted_leads"] == 1


def test_summarize_experiment_rows_marks_empty_metrics_as_no_data():
    rows = [
        StrategyExecutionRecord(
            strategy_name="web_exact_city",
            github_username="grace",
            candidate_name="Grace Hopper",
            cohort_kind="primary",
            cohort_bucket="easy_exact_name",
            aborted=True,
            abort_reason="blocker rate exceeded 10%",
            blocker_state="aborted",
        ),
    ]

    summary = summarize_experiment_rows(rows)
    metrics = summary["strategies"]["web_exact_city"]["aggregate_metrics"]

    assert metrics["row_count"] == 0
    assert metrics["has_data"] is False
    assert metrics["no_candidate_rate"] == 0.0
