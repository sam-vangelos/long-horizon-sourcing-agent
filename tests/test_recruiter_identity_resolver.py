import asyncio
import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from github.reconciliation_input import (
    GitHubReconciliationLead,
    load_saved_github_reconciliation_batch_with_fallback,
)
from linkedin.recruiter_identity_resolver import (
    RecruiterIdentityResolver,
    RecruiterResolverConfig,
)
from shared.reconciliation_schemas import LinkedInIdentityHints, LinkedInMatchResult
from shared.recruiter_identity_schemas import RecruiterIdentityCandidate
from shared.schemas import CandidateProfileSummary, OpusDecision


def _make_lead() -> GitHubReconciliationLead:
    hints = LinkedInIdentityHints(
        candidate_name="Ada Lovelace",
        github_username="ada",
        github_url="https://github.com/ada",
        company="JPMorgan Chase",
        location="New York",
        title="Head of AI Platform",
    )
    return GitHubReconciliationLead(
        username="ada",
        candidate_name="Ada Lovelace",
        github_url="https://github.com/ada",
        company="JPMorgan Chase",
        location="New York",
        title="Head of AI Platform",
        decision="SAVE",
        confidence=0.94,
        rationale="Strong fit",
        source_query="fde",
        source_channel="code_search",
        linkedin_hints=hints,
    )


def test_load_saved_github_reconciliation_batch_with_fallback_uses_saves(tmp_path):
    candidates = [
        {
            "username": "ada",
            "source_query": "fde",
            "source_strategy": "code_search",
            "synthesized_headline": "Head of AI Platform",
            "user": {
                "username": "ada",
                "name": "Ada Lovelace",
                "profile_url": "https://github.com/ada",
                "company": "JPMorgan Chase",
                "location": "New York",
                "bio": "Head of AI Platform",
            },
            "contact": {},
        }
    ]
    saves = [
        {
            "username": "ada",
            "github_url": "https://github.com/ada",
            "decision": "SAVE",
            "rationale": "Strong fit",
        }
    ]
    (tmp_path / "candidates.jsonl").write_text(
        "\n".join(json.dumps(item) for item in candidates),
        encoding="utf-8",
    )
    (tmp_path / "saves.jsonl").write_text(
        "\n".join(json.dumps(item) for item in saves),
        encoding="utf-8",
    )
    (tmp_path / "outreach.jsonl").write_text("", encoding="utf-8")

    batch = load_saved_github_reconciliation_batch_with_fallback(tmp_path)

    assert batch.stats.leads_loaded == 1
    assert batch.leads[0].candidate_name == "Ada Lovelace"
    assert batch.leads[0].linkedin_hints is not None
    assert batch.leads[0].linkedin_hints.candidate_name == "Ada Lovelace"


def test_resolve_lead_high_confidence_without_profile_open_is_manual_tool_failure():
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "\n".join(
            [
                "Ada Lovelace",
                "Head of AI Platform at JPMorgan Chase",
                "New York · 2nd",
            ]
        ),
        "name": "Ada Lovelace",
        "url": "/talent/profile/ada",
        "already_saved": False,
        "recruiter_activity": {"message_count": 1, "project_count": 0, "view_count": 1},
    }
    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(max_cards=3, open_profile_on_likely_match=False),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))

    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.final_action == "MANUAL_REVIEW"
    assert result.final_subreason == "tool_failure"
    assert result.identity_classification == "high_confidence_match"
    # Multi-query lookup: the resolver issues at least one search before early-
    # exiting on the first high-confidence match, and the bounded query plan
    # always carries the synthesized lookup name as a fallback.
    issued_queries = [call.args[0] for call in browser.enter_search_string.await_args_list]
    assert issued_queries, "expected at least one Recruiter search query"
    assert result.queries_tried, "expected queries_tried to record the bounded plan"
    assert "Ada Lovelace" in result.lookup_name
    browser.focus_card_for_review.assert_awaited_with(0)


def test_resolve_lead_returns_manual_review_for_plausible_but_unconfirmed_card():
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "\n".join(
            [
                "Ada Lovelace",
                "Engineering Leader",
                "San Francisco · 3rd",
            ]
        ),
        "name": "Ada Lovelace",
        "url": "/talent/profile/ada2",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=True),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search(""))

    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.final_action == "MANUAL_REVIEW"
    assert result.final_subreason == "identity_ambiguous"
    browser.open_profile_by_url.assert_not_awaited()


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_resolve_lead_opens_sole_initialized_surname_variant_for_identity_confirmation(
    mock_extract,
    mock_judge,
):
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "\n".join(
            [
                "Eri B.",
                "Full-stack Developer at Plastic Labs",
                "New York · 3rd",
            ]
        ),
        "name": "Eri B.",
        "url": "/talent/profile/eri-b",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    browser.get_profile_status_summary.return_value = {}
    mock_extract.return_value = CandidateProfileSummary(
        name="Eri Barrett",
        profile_url="/talent/profile/eri-b",
        headline="Full-stack Developer",
    )
    mock_judge.return_value = OpusDecision(
        stage="full",
        decision="REJECT",
        path="x",
        confidence=0.2,
        rationale="fit fail",
        candidate_name="Eri Barrett",
        profile_url="/talent/profile/eri-b",
    )

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=True, dry_run_save=True),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))

    lead = GitHubReconciliationLead(
        username="erosika",
        candidate_name="Eri Barrett",
        github_url="https://github.com/erosika",
        company="",
        location="NYC",
        title="agentic systems builder",
        decision="SAVE",
        confidence=0.9,
        rationale="",
        source_query="fde",
        source_channel="code_search",
        linkedin_hints=LinkedInIdentityHints(
            candidate_name="Eri Barrett",
            github_username="erosika",
            github_url="https://github.com/erosika",
            company="",
            location="NYC",
            title="agentic systems builder",
        ),
    )

    result = asyncio.run(resolver.resolve_lead(lead))

    assert result.identity_classification == "single_surface_name_variant_profile"
    assert result.opened_profile is True
    assert result.selected_profile_url == "/talent/profile/eri-b"
    browser.open_profile_by_url.assert_awaited_once_with("/talent/profile/eri-b")


def test_resolve_lead_does_not_open_sole_partial_name_without_initial_anchor():
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "\n".join(
            [
                "Ada Byron",
                "Engineering Leader",
                "San Francisco · 3rd",
            ]
        ),
        "name": "Ada Byron",
        "url": "/talent/profile/ada-byron",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=True),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search(""))

    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.identity_classification == "no_confident_match"
    assert result.opened_profile is False
    browser.open_profile_by_url.assert_not_awaited()


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_resolve_lead_save_path_triggers_recruiter_save(mock_extract, mock_judge):
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "\n".join(
            [
                "Ada Lovelace",
                "Head of AI Platform at JPMorgan Chase",
                "New York · 2nd",
            ]
        ),
        "name": "Ada Lovelace",
        "url": "/talent/profile/ada",
        "already_saved": False,
        "recruiter_activity": {"message_count": 1, "project_count": 0, "view_count": 1},
    }
    browser.get_profile_status_summary.return_value = {
        "message_count": 1,
        "project_count": 0,
        "view_count": 1,
        "saved_by": "",
        "last_outbound_contact": "",
    }
    browser.save_candidate.return_value = True

    summary = CandidateProfileSummary(
        name="Ada Lovelace",
        profile_url="/talent/profile/ada",
        headline="Head of AI Platform",
    )
    mock_extract.return_value = summary
    mock_judge.return_value = OpusDecision(
        stage="full",
        decision="SAVE",
        path="DIRECT:1.Test",
        confidence=0.9,
        rationale="Strong",
        candidate_name="Ada Lovelace",
        profile_url="/talent/profile/ada",
    )

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=True, dry_run_save=False),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))

    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.final_action == "SAVE"
    assert result.recruiter_save_attempted is True
    assert result.recruiter_save_succeeded is True
    assert browser.open_profile_by_url.await_count == 2
    browser.open_profile_by_url.assert_awaited_with("/talent/profile/ada")
    browser.simulate_profile_read.assert_awaited_once()
    browser.save_candidate.assert_awaited_once()
    assert browser.go_back_to_results.await_count == 2


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_resolve_lead_dry_run_skips_save_click(mock_extract, mock_judge):
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "Ada Lovelace\nHead of AI Platform at JPMorgan Chase\nNew York · 2nd",
        "name": "Ada Lovelace",
        "url": "/talent/profile/ada",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    browser.get_profile_status_summary.return_value = {}
    mock_extract.return_value = CandidateProfileSummary(
        name="Ada Lovelace", profile_url="/talent/profile/ada", headline="x"
    )
    mock_judge.return_value = OpusDecision(
        stage="full",
        decision="SAVE",
        path="DIRECT:1.Test",
        confidence=0.9,
        rationale="Strong",
        candidate_name="Ada",
        profile_url="/talent/profile/ada",
    )

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=True, dry_run_save=True),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))
    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.final_action == "SAVE"
    browser.save_candidate.assert_not_awaited()


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_resolve_lead_fit_reject_skips_save(mock_extract, mock_judge):
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "Ada Lovelace\nHead of AI Platform at JPMorgan Chase\nNew York · 2nd",
        "name": "Ada Lovelace",
        "url": "/talent/profile/ada",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    browser.get_profile_status_summary.return_value = {}
    mock_extract.return_value = CandidateProfileSummary(
        name="Ada Lovelace", profile_url="/talent/profile/ada", headline="x"
    )
    mock_judge.return_value = OpusDecision(
        stage="full",
        decision="REJECT",
        path="none",
        confidence=0.2,
        rationale="Weak",
        candidate_name="Ada",
        profile_url="/talent/profile/ada",
    )

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=True),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))
    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.final_action == "REJECT"
    assert result.final_subreason == "fit_reject"
    browser.save_candidate.assert_not_awaited()


def _card_snapshot(url: str, rank_suffix: str = "2nd") -> dict:
    return {
        "innertext": "\n".join(
            [
                "Ada Lovelace",
                "Head of AI Platform at JPMorgan Chase",
                f"New York · {rank_suffix}",
            ]
        ),
        "name": "Ada Lovelace",
        "url": url,
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_multi_plausible_two_confirmed_identities_is_manual_review(mock_extract, mock_judge):
    """Plan §5: when more than one opened profile is identity-confirmed, the
    legacy "exactly one SAVE wins" rule is no longer the primary ambiguity
    resolver. The row must be MANUAL_REVIEW with multiple_confirmed_identities,
    even if exactly one of the two profiles fit-rejects."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 2
    browser.get_card_snapshot.side_effect = [
        _card_snapshot("/talent/profile/p1", "2nd"),
        _card_snapshot("/talent/profile/p2", "1st"),
    ]
    browser.get_profile_status_summary.return_value = {}

    def _extract(_innertext: str, profile_url: str):
        # Both profiles look like the same person ("Ada Lovelace") on the open.
        return CandidateProfileSummary(name="Ada Lovelace", profile_url=profile_url, headline="h")

    mock_extract.side_effect = _extract
    mock_judge.side_effect = [
        OpusDecision(
            stage="full",
            decision="REJECT",
            path="none",
            confidence=0.3,
            rationale="no",
            candidate_name="Ada",
            profile_url="/talent/profile/p1",
        ),
        OpusDecision(
            stage="full",
            decision="SAVE",
            path="DIRECT:1.Test",
            confidence=0.9,
            rationale="yes",
            candidate_name="Ada",
            profile_url="/talent/profile/p2",
        ),
    ]
    browser.save_candidate.return_value = True

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            open_profile_on_likely_match=True,
            max_cards=3,
            max_ambiguity_profiles=3,
        ),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))
    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.ambiguity_multi_review is True
    assert len(result.plausible_profile_reviews) == 2
    # Both opened profiles should be marked confirmed by the post-open helper.
    assert result.plausible_profile_reviews[0].identity_status == "confirmed"
    assert result.plausible_profile_reviews[1].identity_status == "confirmed"
    assert result.final_action == "MANUAL_REVIEW"
    assert result.final_subreason == "multiple_confirmed_identities"
    assert result.identity_status == "ambiguous"
    assert result.identity_subreason == "multiple_confirmed_identities"
    # The row must not select either profile when identity is ambiguous.
    assert result.selected_candidate_rank == 0
    assert result.selected_profile_url == ""
    browser.save_candidate.assert_not_awaited()


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_multi_plausible_two_saves_is_manual_ambiguous(mock_extract, mock_judge):
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 2
    browser.get_card_snapshot.side_effect = [
        _card_snapshot("/talent/profile/p1"),
        _card_snapshot("/talent/profile/p2", "1st"),
    ]
    browser.get_profile_status_summary.return_value = {}
    mock_extract.side_effect = lambda _t, profile_url: CandidateProfileSummary(
        name="Ada Lovelace", profile_url=profile_url, headline="h"
    )
    save = OpusDecision(
        stage="full",
        decision="SAVE",
        path="DIRECT:1.A",
        confidence=0.9,
        rationale="ok",
        candidate_name="Ada",
        profile_url="",
    )
    mock_judge.side_effect = [save, save]

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=True, max_cards=3),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))
    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.ambiguity_multi_review is True
    # Plan §5: two identity-confirmed profiles surface as
    # multiple_confirmed_identities, regardless of how many fit-saved.
    assert result.final_action == "MANUAL_REVIEW"
    assert result.final_subreason == "multiple_confirmed_identities"
    assert len(result.plausible_profile_reviews) == 2
    assert result.had_plausible_cards is True
    assert result.selected_candidate_rank == 0
    assert result.selected_profile_url == ""
    assert result.holistic_fit_decision == ""
    browser.save_candidate.assert_not_awaited()


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_multi_ambiguity_unresolved_clears_row_holistic_fields(mock_extract, mock_judge):
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 2
    browser.get_card_snapshot.side_effect = [
        _card_snapshot("/talent/profile/p1"),
        _card_snapshot("/talent/profile/p2", "1st"),
    ]
    browser.get_profile_status_summary.return_value = {}
    mock_extract.side_effect = lambda _t, profile_url: CandidateProfileSummary(
        name="Ada Lovelace", profile_url=profile_url, headline="h"
    )
    borderline = OpusDecision(
        stage="full",
        decision="INFERENTIAL_SAVE",
        path="x",
        confidence=0.55,
        rationale="borderline",
        candidate_name="Ada",
        profile_url="",
    )
    mock_judge.side_effect = [borderline, borderline]

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=True, max_cards=3),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))
    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.ambiguity_multi_review is True
    # Plan §5: identity-first ordering. Two identity-confirmed profiles surface
    # as multiple_confirmed_identities even when both judge as borderline; the
    # row is not selected.
    assert result.final_subreason == "multiple_confirmed_identities"
    assert result.had_plausible_cards is True
    assert result.selected_candidate_rank == 0
    assert result.selected_profile_url == ""
    assert result.holistic_fit_decision == ""
    assert len(result.plausible_profile_reviews) == 2


@patch("linkedin.recruiter_identity_resolver.choose_best_match")
@patch("linkedin.recruiter_identity_resolver.single_plausible_is_safely_dominant", return_value=True)
@patch("linkedin.recruiter_identity_resolver.is_single_strong_plausible_for_profile_open", return_value=True)
@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_single_strong_plausible_opens_profile_when_gates_allow(
    mock_extract,
    mock_judge,
    _mock_is_strong,
    _mock_dom,
    mock_choose,
):
    mock_choose.side_effect = lambda matches: (
        "manual_review",
        sorted(matches, key=lambda m: m.match_confidence, reverse=True)[0],
    )
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 2
    browser.get_card_snapshot.side_effect = [
        {
            "innertext": "Bob Smith\nEngineer at OtherCo\nLondon · 2nd",
            "name": "Bob Smith",
            "url": "/talent/profile/bob",
            "already_saved": False,
            "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
        },
        _card_snapshot("/talent/profile/ada-strong", "1st"),
    ]
    browser.get_profile_status_summary.return_value = {}
    mock_extract.return_value = CandidateProfileSummary(
        name="Ada Lovelace", profile_url="/talent/profile/ada-strong", headline="h"
    )
    mock_judge.return_value = OpusDecision(
        stage="full",
        decision="INFERENTIAL_SAVE",
        path="x",
        confidence=0.55,
        rationale="borderline",
        candidate_name="Ada",
        profile_url="/talent/profile/ada-strong",
    )

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=True, max_cards=3),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))
    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.identity_classification == "single_strong_plausible_profile"
    assert result.opened_profile is True
    assert len(result.plausible_profile_reviews) == 1
    browser.open_profile_by_url.assert_awaited()


@patch("linkedin.recruiter_identity_resolver.choose_best_match")
@patch("linkedin.recruiter_identity_resolver.single_plausible_is_safely_dominant", return_value=False)
@patch("linkedin.recruiter_identity_resolver.is_single_strong_plausible_for_profile_open", return_value=True)
def test_single_strong_path_skipped_when_not_dominant_vs_next_card(
    _mock_strong,
    _mock_dom,
    mock_choose,
):
    mock_choose.side_effect = lambda matches: (
        "manual_review",
        sorted(matches, key=lambda m: m.match_confidence, reverse=True)[0],
    )
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 2
    browser.get_card_snapshot.side_effect = [
        {
            "innertext": "Bob Smith\nEngineer at OtherCo\nLondon · 2nd",
            "name": "Bob Smith",
            "url": "/talent/profile/bob",
            "already_saved": False,
            "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
        },
        _card_snapshot("/talent/profile/ada-strong", "1st"),
    ]

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=True, max_cards=3),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))
    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    browser.open_profile_by_url.assert_not_awaited()
    assert result.opened_profile is False


@patch("linkedin.recruiter_identity_resolver.score_linkedin_identity_match")
@patch("linkedin.recruiter_identity_resolver.choose_best_match")
@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_resolve_anchor_single_plausible_live_shape_opens_without_strong_mocks(
    mock_extract,
    mock_judge,
    mock_choose,
    mock_score,
):
    """End-to-end: tier-2 anchor (0.69 + exact name + company + title) reaches profile open."""

    def _fake_score(_hints, **kwargs):
        url = str(kwargs.get("matched_profile_url") or "")
        if "decoy" in url:
            return LinkedInMatchResult(
                matched_profile_url=url,
                matched_name="Bob Smith",
                match_confidence=0.38,
                evidence=[],
                ambiguity_reasons=["Name mismatch"],
            )
        return LinkedInMatchResult(
            matched_profile_url=url,
            matched_name="Ada Lovelace",
            match_confidence=0.69,
            evidence=["Exact name match", "Company overlap", "Title overlap"],
            ambiguity_reasons=["Location mismatch"],
        )

    mock_score.side_effect = _fake_score
    mock_choose.side_effect = lambda matches: (
        "manual_review",
        sorted(matches, key=lambda m: m.match_confidence, reverse=True)[0],
    )
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 2
    browser.get_card_snapshot.side_effect = [
        {
            "innertext": "Bob Smith\nEngineer at OtherCo\nLondon · 2nd",
            "name": "Bob Smith",
            "url": "/talent/profile/decoy-bob",
            "already_saved": False,
            "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
        },
        _card_snapshot("/talent/profile/ada-anchor", "1st"),
    ]
    browser.get_profile_status_summary.return_value = {}
    mock_extract.return_value = CandidateProfileSummary(
        name="Ada Lovelace", profile_url="/talent/profile/ada-anchor", headline="h"
    )
    mock_judge.return_value = OpusDecision(
        stage="full",
        decision="INFERENTIAL_SAVE",
        path="x",
        confidence=0.55,
        rationale="borderline",
        candidate_name="Ada",
        profile_url="/talent/profile/ada-anchor",
    )

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=True, max_cards=3),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))
    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.identity_classification == "single_strong_plausible_profile"
    assert result.opened_profile is True
    assert len(result.plausible_profile_reviews) == 1
    browser.open_profile_by_url.assert_awaited()


def test_resolve_lead_uses_username_derived_surname_when_candidate_name_is_single_token():
    """P0 fallback: "Michael" + github_username="mldangelo" must search
    "Michael Mldangelo" rather than the bare single-token "Michael"."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 0
    browser.get_card_count.return_value = 0
    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(max_cards=3, open_profile_on_likely_match=False),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York City Metropolitan Area"))

    lead = GitHubReconciliationLead(
        username="mldangelo",
        candidate_name="Michael",
        github_url="https://github.com/mldangelo",
        company="@promptfoo",
        location="New York, NY",
        title="VP Engineering",
        decision="SAVE",
        confidence=0.9,
        rationale="",
        source_query="fde",
        source_channel="code_search",
        linkedin_hints=LinkedInIdentityHints(
            candidate_name="Michael",
            github_username="mldangelo",
            github_url="https://github.com/mldangelo",
            company="@promptfoo",
            location="New York, NY",
            title="VP Engineering",
        ),
    )

    result = asyncio.run(resolver.resolve_lead(lead))

    assert result.lookup_name == "Michael Mldangelo"
    assert result.query == "Michael Mldangelo"
    issued_queries = [call.args[0] for call in browser.enter_search_string.await_args_list]
    # The bounded query plan must include the synthesized "Michael Mldangelo"
    # lookup name (single-token-from-username surname rescue) as a fallback so
    # that downstream operators can see the protected behavior fired.
    assert any("Michael Mldangelo" in q for q in issued_queries), issued_queries
    assert any("Michael Mldangelo" in q for q in result.queries_tried), result.queries_tried
    # No cards surfaced, so the resolver terminates with no_results; this confirms the
    # query string was issued before the "no results" path and was not silently
    # rewritten downstream.
    assert result.identity_classification == "no_results"


def test_multi_query_lookup_recovers_when_first_query_misses(monkeypatch):
    """First bounded query surfaces nothing; a later query in the plan finds the
    profile. The resolver must not give up after the first miss (plan §3)."""
    browser = AsyncMock()
    miss_snapshot = {
        "innertext": "Other Person\nUnrelated\nElsewhere",
        "name": "Other Person",
        "url": "/talent/profile/other",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    hit_snapshot = _card_snapshot("/talent/profile/ada-late", "1st")
    snapshots = {
        # First (combined) query: empty result set
        # Subsequent queries: at least one returns the strong hit
    }
    # Drive different snapshots based on query order: first query returns miss, second returns hit.
    queries_seen: list[str] = []

    async def _enter_search(query: str) -> None:
        queries_seen.append(query)

    browser.enter_search_string.side_effect = _enter_search

    async def _slot_count() -> int:
        # Both queries yield 1 card slot.
        return 1

    browser.get_card_slot_count.side_effect = _slot_count
    browser.get_card_count.return_value = 1

    async def _snapshot(_index: int) -> dict:
        # The first query gets the miss, all later queries get the hit.
        return miss_snapshot if len(queries_seen) <= 1 else hit_snapshot

    browser.get_card_snapshot.side_effect = _snapshot

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=False, max_cards=2),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))

    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert len(queries_seen) >= 2, queries_seen
    surfaced_urls = [c.profile_url for c in result.top_candidates]
    assert "/talent/profile/ada-late" in surfaced_urls
    assert result.queries_tried, "expected the bounded query plan to be recorded"
    assert result.resolved_query in result.queries_tried, (
        result.resolved_query,
        result.queries_tried,
    )


def test_direct_linkedin_hint_anchors_surfaced_card():
    """When GitHub hints carry a direct LinkedIn URL and a Recruiter card surfaces
    with that URL, the resolver must annotate the card as a direct-hint identity
    anchor (plan §3)."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = _card_snapshot("/talent/profile/ada-direct", "1st")

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(open_profile_on_likely_match=False, max_cards=2),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))

    hints = LinkedInIdentityHints(
        candidate_name="Ada Lovelace",
        github_username="ada",
        github_url="https://github.com/ada",
        linkedin_url_hint="/talent/profile/ada-direct",
        company="JPMorgan Chase",
        location="New York",
        title="Head of AI Platform",
    )
    lead = GitHubReconciliationLead(
        username="ada",
        candidate_name="Ada Lovelace",
        github_url="https://github.com/ada",
        company="JPMorgan Chase",
        location="New York",
        title="Head of AI Platform",
        decision="SAVE",
        confidence=0.9,
        rationale="",
        source_query="fde",
        source_channel="code_search",
        linkedin_hints=hints,
    )

    result = asyncio.run(resolver.resolve_lead(lead))

    surfaced = [c for c in result.top_candidates if c.profile_url == "/talent/profile/ada-direct"]
    assert surfaced, result.top_candidates
    evidence_blob = " | ".join(surfaced[0].evidence)
    assert "Direct LinkedIn URL hint" in evidence_blob, evidence_blob


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_identity_collect_confirmed_identity_collects_without_full_judge(
    mock_extract,
    mock_judge,
):
    """In identity_collect, a confirmed Recruiter identity must collect without
    invoking full_judge (plan §4)."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "Ada Lovelace\nHead of AI Platform at JPMorgan Chase\nNew York · 2nd",
        "name": "Ada Lovelace",
        "url": "/talent/profile/ada",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    browser.get_profile_status_summary.return_value = {}
    mock_extract.return_value = CandidateProfileSummary(
        name="Ada Lovelace",
        profile_url="/talent/profile/ada",
        headline="Head of AI Platform",
    )
    browser.save_candidate.return_value = True

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            open_profile_on_likely_match=True,
            workflow_mode="identity_collect",
        ),
        linkedin_brief=None,
    )
    asyncio.run(resolver.prepare_search("New York"))

    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    mock_judge.assert_not_called()
    assert result.workflow_mode == "identity_collect"
    assert result.identity_status == "confirmed"
    assert result.collection_action == "COLLECT"
    assert result.project_save_state == "saved_now"
    assert result.opened_profile is True
    browser.save_candidate.assert_awaited_once()


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_identity_collect_already_saved_confirmed_identity_does_not_attempt_save(
    mock_extract,
    mock_judge,
):
    """Already-saved confirmed identities still collect with annotation-only save state."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "Ada Lovelace\nHead of AI Platform at JPMorgan Chase\nNew York · 2nd",
        "name": "Ada Lovelace",
        "url": "/talent/profile/ada",
        "already_saved": True,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    browser.get_profile_status_summary.return_value = {}
    mock_extract.return_value = CandidateProfileSummary(
        name="Ada Lovelace",
        profile_url="/talent/profile/ada",
        headline="Head of AI Platform",
    )

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            open_profile_on_likely_match=True,
            workflow_mode="identity_collect",
        ),
        linkedin_brief=None,
    )
    asyncio.run(resolver.prepare_search("New York"))

    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    mock_judge.assert_not_called()
    assert result.collection_action == "COLLECT"
    assert result.project_save_state == "already_saved"
    browser.save_candidate.assert_not_awaited()


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_identity_collect_dry_run_skips_save_click_but_still_collects(
    mock_extract,
    mock_judge,
):
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "Ada Lovelace\nHead of AI Platform at JPMorgan Chase\nNew York · 2nd",
        "name": "Ada Lovelace",
        "url": "/talent/profile/ada",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    browser.get_profile_status_summary.return_value = {}
    mock_extract.return_value = CandidateProfileSummary(
        name="Ada Lovelace",
        profile_url="/talent/profile/ada",
        headline="Head of AI Platform",
    )

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            open_profile_on_likely_match=True,
            workflow_mode="identity_collect",
            dry_run_save=True,
        ),
        linkedin_brief=None,
    )
    asyncio.run(resolver.prepare_search("New York"))

    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    mock_judge.assert_not_called()
    assert result.collection_action == "COLLECT"
    assert result.project_save_state == "dry_run_skipped"
    browser.save_candidate.assert_not_awaited()


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_identity_collect_partial_name_wrong_person_does_not_collect(
    mock_extract,
    mock_judge,
):
    """When the opened profile contradicts GitHub identity, identity_collect must
    NOT mark the row as COLLECT (plan §4)."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "Eri B.\nFull-stack Developer at Plastic Labs\nNew York · 3rd",
        "name": "Eri B.",
        "url": "/talent/profile/eri-b",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    browser.get_profile_status_summary.return_value = {}
    # The opened profile turns out to be a different person ("Erica Bonilla"),
    # NOT Eri Barrett. Identity must not be confirmed.
    mock_extract.return_value = CandidateProfileSummary(
        name="Erica Bonilla",
        profile_url="/talent/profile/eri-b",
        headline="Different person at Different Co",
    )

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            open_profile_on_likely_match=True,
            workflow_mode="identity_collect",
        ),
        linkedin_brief=None,
    )
    asyncio.run(resolver.prepare_search("New York"))

    lead = GitHubReconciliationLead(
        username="erosika",
        candidate_name="Eri Barrett",
        github_url="https://github.com/erosika",
        company="",
        location="NYC",
        title="agentic systems builder",
        decision="SAVE",
        confidence=0.9,
        rationale="",
        source_query="fde",
        source_channel="code_search",
        linkedin_hints=LinkedInIdentityHints(
            candidate_name="Eri Barrett",
            github_username="erosika",
            github_url="https://github.com/erosika",
            company="",
            location="NYC",
            title="agentic systems builder",
        ),
    )

    result = asyncio.run(resolver.resolve_lead(lead))

    mock_judge.assert_not_called()
    assert result.collection_action != "COLLECT", result.collection_action
    assert result.project_save_state == "not_attempted"
    browser.save_candidate.assert_not_awaited()


def test_compute_post_open_identity_status_exact_normalized_name_is_confirmed():
    """Pure helper: exact normalized profile name match → confirmed (plan §4)."""
    from linkedin.recruiter_identity_resolver import compute_post_open_identity_status

    hints = LinkedInIdentityHints(candidate_name="Ada Lovelace")
    candidate = RecruiterIdentityCandidate(rank=1, name="Ada L.", profile_url="/p")
    profile_summary = CandidateProfileSummary(
        name="Ada Lovelace",
        profile_url="/p",
        headline="Engineer",
    )

    status, subreason = compute_post_open_identity_status(
        hints=hints,
        candidate=candidate,
        profile_summary=profile_summary,
        extraction_failed=False,
    )

    assert status == "confirmed"
    assert subreason == "exact_normalized_profile_name_match"


def test_compute_post_open_identity_status_extraction_failed_is_tool_failure():
    from linkedin.recruiter_identity_resolver import compute_post_open_identity_status

    hints = LinkedInIdentityHints(candidate_name="Ada Lovelace")
    candidate = RecruiterIdentityCandidate(rank=1, name="Ada Lovelace", profile_url="/p")

    status, subreason = compute_post_open_identity_status(
        hints=hints,
        candidate=candidate,
        profile_summary=None,
        extraction_failed=True,
    )

    assert status == "tool_failure"
    assert subreason == "extraction_failed"


def test_compute_post_open_identity_status_no_overlap_is_no_match():
    from linkedin.recruiter_identity_resolver import compute_post_open_identity_status

    hints = LinkedInIdentityHints(candidate_name="Ada Lovelace")
    candidate = RecruiterIdentityCandidate(rank=1, name="Ada L.", profile_url="/p")
    profile_summary = CandidateProfileSummary(
        name="Bob Smith",
        profile_url="/p",
        headline="Other person",
    )

    status, subreason = compute_post_open_identity_status(
        hints=hints,
        candidate=candidate,
        profile_summary=profile_summary,
        extraction_failed=False,
    )

    assert status == "no_match"
    assert subreason == "profile_name_contradicts_github_identity"


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_multi_profile_one_confirmed_one_fit_better_non_match_does_not_pick_by_fit(
    mock_extract,
    mock_judge,
):
    """Plan §5 protected case: rank-1 card opens to a wrong-person profile that
    happens to score SAVE on fit; rank-2 card opens to the correct identity but
    judges REJECT. Identity-first ordering MUST pick the confirmed identity and
    must NOT route through fit. fit_gated_save then runs the fit gate ON the
    confirmed profile (which says REJECT here) and produces REJECT, not SAVE."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 2
    browser.get_card_snapshot.side_effect = [
        # Rank-1 surface: looks plausible (Ada Lovelace name)
        _card_snapshot("/talent/profile/wrong-fit-better", "1st"),
        # Rank-2 surface: also looks plausible (Ada Lovelace name)
        _card_snapshot("/talent/profile/right-identity", "2nd"),
    ]
    browser.get_profile_status_summary.return_value = {}

    def _extract(_innertext: str, profile_url: str):
        # Rank-1 opens to a different person on the LinkedIn profile (wrong identity)
        if profile_url == "/talent/profile/wrong-fit-better":
            return CandidateProfileSummary(
                name="Bob Smith",
                profile_url=profile_url,
                headline="Different person — happens to fit the brief",
            )
        # Rank-2 opens to the right person (Ada Lovelace)
        return CandidateProfileSummary(
            name="Ada Lovelace",
            profile_url=profile_url,
            headline="The actual GitHub lead",
        )

    mock_extract.side_effect = _extract

    # Followups plan §3: full_judge is now only called for identity-confirmed
    # profiles. We key the judge stub by profile_url so the test still proves
    # that the fit gate runs ON the confirmed card (and would NOT save even if
    # the wrong-identity card had a SAVE-shaped fit decision).
    def _judge(profile_summary, brief=None):
        if profile_summary.profile_url == "/talent/profile/wrong-fit-better":
            return OpusDecision(
                stage="full",
                decision="SAVE",
                path="DIRECT:1.Wrong",
                confidence=0.9,
                rationale="would have fit-saved if we asked",
                candidate_name="Bob Smith",
                profile_url="/talent/profile/wrong-fit-better",
            )
        return OpusDecision(
            stage="full",
            decision="REJECT",
            path="none",
            confidence=0.2,
            rationale="not a fit for current brief",
            candidate_name="Ada Lovelace",
            profile_url="/talent/profile/right-identity",
        )

    mock_judge.side_effect = _judge
    browser.save_candidate.return_value = True

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            open_profile_on_likely_match=True,
            max_cards=3,
            max_ambiguity_profiles=3,
            workflow_mode="fit_gated_save",
        ),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))
    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.ambiguity_multi_review is True
    # Identity-first: only the rank-2 (right-identity) profile is confirmed.
    confirmed = [r for r in result.plausible_profile_reviews if r.identity_status == "confirmed"]
    assert len(confirmed) == 1, [r.identity_status for r in result.plausible_profile_reviews]
    assert confirmed[0].profile_url == "/talent/profile/right-identity"
    # The fit gate runs only on the confirmed profile (REJECT), so the row
    # rejects rather than saving the wrong-identity-but-fit-better card.
    assert result.final_action == "REJECT", (result.final_action, result.final_subreason)
    assert result.selected_profile_url == "/talent/profile/right-identity"
    browser.save_candidate.assert_not_awaited()
    # Followups plan §3: full_judge must NEVER be called on the unconfirmed
    # wrong-identity card. Verify by inspecting the actual full_judge call
    # arguments — only the right-identity profile_url should appear.
    judged_urls = [
        call.args[0].profile_url for call in mock_judge.call_args_list
    ]
    assert judged_urls == ["/talent/profile/right-identity"], judged_urls


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_fit_gated_save_unconfirmed_opened_profile_does_not_call_full_judge(
    mock_extract,
    mock_judge,
):
    """Followups plan §3: in fit_gated_save, when post-open identity is NOT
    confirmed, full_judge MUST NOT be called. The row routes through the
    synthetic identity-only terminal."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "Eri B.\nFull-stack Developer at Plastic Labs\nNew York · 3rd",
        "name": "Eri B.",
        "url": "/talent/profile/eri-b",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    browser.get_profile_status_summary.return_value = {}
    # Opened profile turns out to be a different person — identity should NOT
    # confirm (no normalized name match, no direct hint, no card-name+structural
    # overlap from this lead).
    mock_extract.return_value = CandidateProfileSummary(
        name="Erica Bonilla",
        profile_url="/talent/profile/eri-b",
        headline="Different person at Different Co",
    )

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            open_profile_on_likely_match=True,
            workflow_mode="fit_gated_save",
        ),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))

    lead = GitHubReconciliationLead(
        username="erosika",
        candidate_name="Eri Barrett",
        github_url="https://github.com/erosika",
        company="",
        location="NYC",
        title="agentic systems builder",
        decision="SAVE",
        confidence=0.9,
        rationale="",
        source_query="fde",
        source_channel="code_search",
        linkedin_hints=LinkedInIdentityHints(
            candidate_name="Eri Barrett",
            github_username="erosika",
            github_url="https://github.com/erosika",
            company="",
            location="NYC",
            title="agentic systems builder",
        ),
    )

    result = asyncio.run(resolver.resolve_lead(lead))

    # full_judge must NEVER have been called.
    mock_judge.assert_not_called()
    # The row routes through the identity-only terminal.
    assert result.opened_profile is True
    assert result.identity_status in ("no_match", "ambiguous"), result.identity_status
    assert result.final_action in ("MANUAL_REVIEW", "REJECT"), result.final_action
    # And holistic-fit fields stay empty (we never asked the judge).
    assert result.holistic_fit_decision == ""
    assert result.holistic_fit_path == ""


def test_identity_collect_no_results_row_populates_canonical_fields():
    """Followups plan §4: when no Recruiter cards surface, identity_collect rows
    must still set canonical identity_status / collection_action /
    project_save_state."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 0
    browser.get_card_count.return_value = 0

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            open_profile_on_likely_match=True,
            workflow_mode="identity_collect",
        ),
        linkedin_brief=None,
    )
    asyncio.run(resolver.prepare_search("New York"))
    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.identity_classification == "no_results"
    assert result.identity_status == "no_match"
    assert result.identity_subreason == "no_recruiter_results_surfaced"
    assert result.collection_action == "REJECT"
    assert result.collection_subreason == "no_recruiter_results"
    assert result.project_save_state == "not_attempted"


def test_identity_collect_no_confident_match_row_populates_canonical_fields():
    """Followups plan §4: surface-level no_confident_match path (single weak
    card with no plausibility floor) populates ambiguous canonical fields."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "Bob Smith\nUnrelated\nElsewhere",
        "name": "Bob Smith",
        "url": "/talent/profile/bob",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            open_profile_on_likely_match=True,
            workflow_mode="identity_collect",
        ),
        linkedin_brief=None,
    )
    asyncio.run(resolver.prepare_search(""))
    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.opened_profile is False
    # Surface-level ambiguity / no-confident-match → canonical "ambiguous" row.
    assert result.identity_status == "ambiguous"
    assert result.collection_action == "MANUAL_REVIEW"
    assert result.project_save_state == "not_attempted"
    assert result.identity_subreason  # non-empty (e.g. "no_confident_match")
    assert result.collection_subreason


def test_identity_collect_high_confidence_without_open_populates_canonical_fields():
    """Followups plan §4: when --skip-profile-open prevents opening, the
    high-confidence card row must still carry canonical fields. The canonical
    action is MANUAL_REVIEW (we never confirmed identity post-open)."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = _card_snapshot("/talent/profile/ada", "1st")

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            max_cards=3,
            open_profile_on_likely_match=False,
            workflow_mode="identity_collect",
        ),
        linkedin_brief=None,
    )
    asyncio.run(resolver.prepare_search("New York"))
    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    assert result.opened_profile is False
    assert result.identity_status == "ambiguous"
    assert result.collection_action == "MANUAL_REVIEW"
    assert result.project_save_state == "not_attempted"


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_identity_collect_public_linkedin_hint_does_not_confirm_via_free_text_innertext(
    mock_extract,
    mock_judge,
):
    """Review-Fixes §3 (regression guard): a public /in/<slug> hint that ONLY
    appears in the opened profile innertext (free text) must NOT confirm
    identity. Once a subject-owned public profile URL extractor lands, this
    test will be paired with a positive test that re-enables the anchor on
    that stronger source.

    Without the anchor, this row falls back to the standard post-open
    identity check; since the opened profile name ("Ada L.") and the GitHub
    name ("Ada Lovelace") share a token but the profile lacks structural
    overlap, the row lands at "ambiguous" instead of "confirmed".
    """
    public_slug = "ada-lovelace"
    profile_text = "\n".join(
        [
            "Ada L.",
            "Some Title at Some Company",
            "Different Location",
            f"Profile: https://www.linkedin.com/in/{public_slug}",
        ]
    )
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "Ada L.\nSome Title at Some Company\nDifferent Location · 3rd",
        "name": "Ada L.",
        "url": "/talent/profile/ada-anchor",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    browser.get_profile_status_summary.return_value = {}
    browser.get_profile_innertext.return_value = profile_text
    mock_extract.return_value = CandidateProfileSummary(
        name="Ada L.",
        profile_url="/talent/profile/ada-anchor",
        headline="Some Title",
    )
    browser.save_candidate.return_value = True

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            open_profile_on_likely_match=True,
            workflow_mode="identity_collect",
            max_cards=2,
            dry_run_save=True,
        ),
        linkedin_brief=None,
    )
    asyncio.run(resolver.prepare_search("New York"))

    lead = GitHubReconciliationLead(
        username="ada",
        candidate_name="Ada Lovelace",
        github_url="https://github.com/ada",
        company="JPMorgan Chase",
        location="New York",
        title="Head of AI Platform",
        decision="SAVE",
        confidence=0.9,
        rationale="",
        source_query="fde",
        source_channel="code_search",
        linkedin_hints=LinkedInIdentityHints(
            candidate_name="Ada Lovelace",
            github_username="ada",
            github_url="https://github.com/ada",
            linkedin_url_hint=f"https://www.linkedin.com/in/{public_slug}",
            company="JPMorgan Chase",
            location="New York",
            title="Head of AI Platform",
        ),
    )

    result = asyncio.run(resolver.resolve_lead(lead))

    mock_judge.assert_not_called()
    # The row must NOT confirm via the disabled public free-text anchor.
    assert result.identity_status != "confirmed", result
    assert result.identity_subreason != "direct_linkedin_hint_anchor", result
    assert result.collection_action != "COLLECT", result.collection_action


def test_normalize_public_linkedin_handle_extracts_lowercase_slug():
    from linkedin.recruiter_identity_resolver import normalize_public_linkedin_handle

    assert normalize_public_linkedin_handle("https://www.linkedin.com/in/Foo-Bar/") == "foo-bar"
    assert normalize_public_linkedin_handle("linkedin.com/in/foo") == "foo"
    assert normalize_public_linkedin_handle("/in/foo") == "foo"
    assert normalize_public_linkedin_handle("https://www.linkedin.com/in/foo?utm=x") == "foo"
    assert normalize_public_linkedin_handle("https://www.linkedin.com/in/foo#about") == "foo"
    # Recruiter URLs do not carry a public handle.
    assert normalize_public_linkedin_handle("/talent/profile/AAA") == ""
    # Empty / unrecognized.
    assert normalize_public_linkedin_handle("") == ""
    assert normalize_public_linkedin_handle("https://github.com/foo") == ""


def test_apply_direct_hint_anchor_only_fires_for_recruiter_talent_urls():
    """Followups plan §5: surface-time anchor must NOT fire for public /in/
    URLs. Public hints are post-open only."""
    from shared.reconciliation_schemas import LinkedInMatchResult

    candidate = RecruiterIdentityCandidate(
        rank=1,
        profile_url="/talent/profile/ada",
        name="Ada Lovelace",
    )
    match = LinkedInMatchResult(
        matched_profile_url="/talent/profile/ada",
        matched_name="Ada Lovelace",
        match_confidence=0.7,
        evidence=[],
        ambiguity_reasons=[],
    )
    hints = LinkedInIdentityHints(
        candidate_name="Ada Lovelace",
        # GitHub-sourced PUBLIC LinkedIn URL hint. Should NOT anchor at the
        # Recruiter card surface (different URL spaces).
        linkedin_url_hint="https://www.linkedin.com/in/ada-lovelace",
    )

    RecruiterIdentityResolver._apply_direct_hint_anchor([(candidate, match)], hints)

    from linkedin.recruiter_identity_resolver import DIRECT_HINT_EVIDENCE

    assert DIRECT_HINT_EVIDENCE not in candidate.evidence, candidate.evidence


def test_apply_direct_hint_anchor_fires_for_recruiter_talent_url_hint():
    """When the hint is itself a Recruiter URL and a card has the same URL,
    the surface-time anchor still fires (preserves the v1 behavior for the
    rare Recruiter-hint case)."""
    from shared.reconciliation_schemas import LinkedInMatchResult

    candidate = RecruiterIdentityCandidate(
        rank=1,
        profile_url="/talent/profile/ada",
        name="Ada Lovelace",
    )
    match = LinkedInMatchResult(
        matched_profile_url="/talent/profile/ada",
        matched_name="Ada Lovelace",
        match_confidence=0.7,
        evidence=[],
        ambiguity_reasons=[],
    )
    hints = LinkedInIdentityHints(
        candidate_name="Ada Lovelace",
        linkedin_url_hint="/talent/profile/ada",
    )

    RecruiterIdentityResolver._apply_direct_hint_anchor([(candidate, match)], hints)

    from linkedin.recruiter_identity_resolver import DIRECT_HINT_EVIDENCE

    assert DIRECT_HINT_EVIDENCE in candidate.evidence


def test_apply_post_open_public_hint_anchor_does_not_match_free_text_slug():
    """Review-Fixes §3: the post-open public-hint anchor is disabled until a
    subject-owned public profile URL is available. An incidental free-text
    /in/<slug> mention must NOT confirm identity, even when the slug matches
    the hint exactly. Surface-time anchoring still works for true Recruiter
    /talent/profile/ hints."""
    candidate = RecruiterIdentityCandidate(
        rank=1,
        profile_url="/talent/profile/ada",
        name="Ada L.",
    )
    hints = LinkedInIdentityHints(
        candidate_name="Ada Lovelace",
        linkedin_url_hint="https://www.linkedin.com/in/ada-lovelace",
    )
    # Even when the slug appears in the profile body (which could be the
    # subject's own URL OR an incidental reference to a co-worker), the helper
    # must NOT anchor — there is no way to distinguish subject-owned vs
    # incidental from free text.
    profile_text = "\n".join(
        [
            "Ada L.",
            "Head of AI Platform at JPMorgan Chase",
            "Contact: https://www.linkedin.com/in/ada-lovelace",
        ]
    )

    matched = RecruiterIdentityResolver._apply_post_open_public_hint_anchor(
        candidate, hints, profile_text
    )

    from linkedin.recruiter_identity_resolver import (
        DIRECT_HINT_EVIDENCE,
        PUBLIC_HINT_EVIDENCE,
    )

    assert matched is False
    assert PUBLIC_HINT_EVIDENCE not in candidate.evidence
    assert DIRECT_HINT_EVIDENCE not in candidate.evidence


def test_apply_post_open_public_hint_anchor_no_match_when_slug_differs():
    """Negative case (still false post-disable): a different /in/<slug> in
    the profile body never anchors."""
    candidate = RecruiterIdentityCandidate(rank=1, profile_url="/talent/profile/ada")
    hints = LinkedInIdentityHints(
        candidate_name="Ada Lovelace",
        linkedin_url_hint="https://www.linkedin.com/in/ada-lovelace",
    )
    # Profile innertext mentions a DIFFERENT public slug (e.g. a co-worker).
    profile_text = "Worked with https://www.linkedin.com/in/bob-smith"

    matched = RecruiterIdentityResolver._apply_post_open_public_hint_anchor(
        candidate, hints, profile_text
    )

    from linkedin.recruiter_identity_resolver import DIRECT_HINT_EVIDENCE

    assert matched is False
    assert DIRECT_HINT_EVIDENCE not in candidate.evidence


def test_apply_post_open_public_hint_anchor_skips_when_hint_is_recruiter_url():
    """Recruiter URL hints are anchored at surface time; the post-open helper
    must not double-anchor on them (and post-disable, never anchors anyway)."""
    candidate = RecruiterIdentityCandidate(rank=1, profile_url="/talent/profile/ada")
    hints = LinkedInIdentityHints(
        candidate_name="Ada Lovelace",
        linkedin_url_hint="/talent/profile/ada",
    )
    profile_text = "https://www.linkedin.com/in/ada-lovelace"

    matched = RecruiterIdentityResolver._apply_post_open_public_hint_anchor(
        candidate, hints, profile_text
    )

    from linkedin.recruiter_identity_resolver import (
        DIRECT_HINT_EVIDENCE,
        PUBLIC_HINT_EVIDENCE,
    )

    assert matched is False
    assert PUBLIC_HINT_EVIDENCE not in candidate.evidence
    assert DIRECT_HINT_EVIDENCE not in candidate.evidence


def test_compute_post_open_identity_status_card_name_plus_structural_overlap_confirms():
    """Card name was a near-exact match (initialized surname) and the opened
    profile shares structural overlap (company OR title OR location) → confirmed."""
    from shared.schemas import Experience
    from linkedin.recruiter_identity_resolver import compute_post_open_identity_status

    hints = LinkedInIdentityHints(
        candidate_name="Eri Barrett",
        company="Plastic Labs",
        title="Full-stack Developer",
        location="New York",
    )
    candidate = RecruiterIdentityCandidate(rank=1, name="Eri B.", profile_url="/p")
    profile_summary = CandidateProfileSummary(
        name="Eri B.",  # opened profile's surface name still abbreviated
        profile_url="/p",
        headline="Full-stack Developer",
        experiences=[
            Experience(title="Full-stack Developer", company="Plastic Labs", location="New York"),
        ],
    )

    status, subreason = compute_post_open_identity_status(
        hints=hints,
        candidate=candidate,
        profile_summary=profile_summary,
        extraction_failed=False,
    )

    assert status == "confirmed"
    assert subreason == "card_name_plus_structural_overlap"


def test_replay_query_for_candidate_returns_rank_when_no_surfaced_query():
    """Candidates without a surfaced_query (single-query lookup, or already on
    the right page) skip replay and use rank-based focus instead."""

    async def _go():
        browser = AsyncMock()
        resolver = RecruiterIdentityResolver(
            browser=browser,
            config=RecruiterResolverConfig(),
            linkedin_brief=None,
        )
        candidate = RecruiterIdentityCandidate(
            rank=3, profile_url="/p", surfaced_query=""
        )
        index = await resolver._replay_query_for_candidate(candidate)
        return browser, index

    browser, index = asyncio.run(_go())
    assert index == 2
    browser.enter_search_string.assert_not_awaited()


def test_replay_query_re_enters_winning_query_and_finds_card_by_url():
    """Followups plan §2: when the candidate's surfaced_query differs from the
    browser's current query, _replay_query_for_candidate must re-enter the
    surfaced_query, scan card snapshots, and return the matching card index."""

    async def _go():
        browser = AsyncMock()
        browser.get_card_slot_count.return_value = 3
        snapshots = {
            0: {"url": "/talent/profile/other-1"},
            1: {"url": "/talent/profile/wanted"},
            2: {"url": "/talent/profile/other-2"},
        }

        async def _snapshot(index: int):
            return snapshots[index]

        browser.get_card_snapshot.side_effect = _snapshot

        resolver = RecruiterIdentityResolver(
            browser=browser,
            config=RecruiterResolverConfig(max_cards=3),
            linkedin_brief=None,
        )
        # Simulate the multi-query loop having ended on a different query.
        resolver._current_browser_query = "later query"
        candidate = RecruiterIdentityCandidate(
            rank=99,
            profile_url="/talent/profile/wanted",
            surfaced_query="winning query",
        )
        index = await resolver._replay_query_for_candidate(candidate)
        return browser, index

    browser, index = asyncio.run(_go())
    assert index == 1, index
    browser.enter_search_string.assert_awaited_once_with("winning query")


def test_replay_query_skips_re_enter_when_already_on_winning_query():
    """No-op replay: if the browser is already on the surfaced_query, do not
    re-enter the query string just to scan for the card index."""

    async def _go():
        browser = AsyncMock()
        browser.get_card_slot_count.return_value = 1
        browser.get_card_snapshot.return_value = {"url": "/talent/profile/ada"}
        resolver = RecruiterIdentityResolver(
            browser=browser,
            config=RecruiterResolverConfig(max_cards=2),
            linkedin_brief=None,
        )
        resolver._current_browser_query = "matching query"
        candidate = RecruiterIdentityCandidate(
            rank=99,
            profile_url="/talent/profile/ada",
            surfaced_query="matching query",
        )
        index = await resolver._replay_query_for_candidate(candidate)
        return browser, index

    browser, index = asyncio.run(_go())
    assert index == 0, index
    browser.enter_search_string.assert_not_awaited()


def test_focus_then_open_candidate_replays_then_opens_by_url():
    """End-to-end of the helper used by save paths: enter the surfacing query,
    focus the matched card, then click open_profile_by_url."""

    async def _go():
        browser = AsyncMock()
        browser.get_card_slot_count.return_value = 1
        browser.get_card_snapshot.return_value = {"url": "/talent/profile/ada"}
        resolver = RecruiterIdentityResolver(
            browser=browser,
            config=RecruiterResolverConfig(max_cards=2),
            linkedin_brief=None,
        )
        resolver._current_browser_query = "different"
        candidate = RecruiterIdentityCandidate(
            rank=42,
            profile_url="/talent/profile/ada",
            surfaced_query="winning",
        )
        await resolver._focus_then_open_candidate(candidate)
        return browser

    browser = asyncio.run(_go())
    browser.enter_search_string.assert_awaited_once_with("winning")
    browser.focus_card_for_review.assert_awaited_once_with(0)
    browser.open_profile_by_url.assert_awaited_once_with("/talent/profile/ada")


def test_use_existing_search_does_not_navigate_or_apply_filters():
    browser = AsyncMock()
    resolver = RecruiterIdentityResolver(
        browser=browser,
        config=RecruiterResolverConfig(),
        linkedin_brief=MagicMock(),
    )

    asyncio.run(resolver.use_existing_search("New York City Metropolitan Area"))

    assert resolver.search_location == "New York City Metropolitan Area"
    browser.navigate_to_search.assert_not_awaited()
    browser.apply_permanent_filters.assert_not_awaited()
    browser.go_back_to_results.assert_awaited_once()


# ---------------------------------------------------------------------------
# Recruiter-Identity-Collection-Cycle-Audit-Fixes — Slice 5 regression tests
# ---------------------------------------------------------------------------


def _eri_barrett_lead() -> GitHubReconciliationLead:
    """Helper: lead shape used by the screenshot-shaped sole-init regression test."""
    return GitHubReconciliationLead(
        username="erosika",
        candidate_name="Eri Barrett",
        github_url="https://github.com/erosika",
        company="Plastic Labs",
        location="New York",
        title="agentic systems builder",
        decision="SAVE",
        confidence=0.9,
        rationale="",
        source_query="fde",
        source_channel="code_search",
        linkedin_hints=LinkedInIdentityHints(
            candidate_name="Eri Barrett",
            github_username="erosika",
            github_url="https://github.com/erosika",
            company="Plastic Labs",
            location="New York",
            title="agentic systems builder",
        ),
    )


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_first_query_sole_initialized_surname_card_stops_before_fallbacks(
    mock_extract,
    mock_judge,
):
    """Recruiter-Identity-Collection-Cycle-Audit-Fixes §3 + §5 (screenshot-shaped):
    in identity_collect + use_current_search, when the first issued query
    surfaces exactly one initialized-surname card, the resolver MUST stop
    before issuing any further title/location queries and must open that card
    for identity confirmation. No fallback enrichment is allowed."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "\n".join(
            [
                "Eri B.",
                "Full-stack Developer at Plastic Labs",
                "New York · 3rd",
            ]
        ),
        "name": "Eri B.",
        "url": "/talent/profile/eri-b",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    browser.get_profile_status_summary.return_value = {}
    mock_extract.return_value = CandidateProfileSummary(
        name="Eri Barrett",
        profile_url="/talent/profile/eri-b",
        headline="Full-stack Developer",
    )

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            open_profile_on_likely_match=True,
            workflow_mode="identity_collect",
            dry_run_save=True,
        ),
        linkedin_brief=None,
    )
    # use_existing_search marks the resolver as attached to a pre-filtered
    # Recruiter search; in "auto" policy this picks name_first.
    asyncio.run(resolver.use_existing_search("New York"))

    result = asyncio.run(resolver.resolve_lead(_eri_barrett_lead()))

    issued_queries = [call.args[0] for call in browser.enter_search_string.await_args_list]
    # Exactly ONE query was issued before the resolver stopped.
    assert len(issued_queries) == 1, issued_queries
    # And it was the bare quoted lookup name — no title/location/company
    # tokens were typed into the operator-prepared search.
    assert issued_queries[0] == '"Eri Barrett"', issued_queries[0]
    # The resolver did open the surfaced card for identity confirmation.
    browser.open_profile_by_url.assert_awaited_with("/talent/profile/eri-b")
    # full_judge is never called in identity_collect.
    mock_judge.assert_not_called()
    # Provenance is honest: queries_tried is the attempted prefix, planned
    # plan is captured separately, stop reason names the new branch.
    assert result.queries_tried == ['"Eri Barrett"']
    assert result.planned_queries  # non-empty
    assert result.stop_reason == "single_surface_name_variant_stop"
    # The post-loop sole-init open branch fired.
    assert result.identity_classification == "single_surface_name_variant_profile"


def test_identity_collect_use_current_search_no_results_does_not_run_enriched_fallback():
    """Cycle-Audit-Fixes §1 + §5: identity_collect with use_current_search and
    no surfaced cards must NOT fall back to enriched company/location/title
    queries. Only the bare-name query was attempted."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 0
    browser.get_card_count.return_value = 0

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            workflow_mode="identity_collect",
            open_profile_on_likely_match=True,
        ),
        linkedin_brief=None,
    )
    asyncio.run(resolver.use_existing_search("New York"))

    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    issued_queries = [call.args[0] for call in browser.enter_search_string.await_args_list]
    assert len(issued_queries) == 1, issued_queries
    assert issued_queries[0] == '"Ada Lovelace"', issued_queries[0]
    assert result.queries_tried == ['"Ada Lovelace"']
    assert result.stop_reason == "plan_exhausted"
    # Canonical identity_collect terminal fields must still be backfilled.
    assert result.identity_classification == "no_results"
    assert result.identity_status == "no_match"
    assert result.collection_action == "REJECT"
    assert result.project_save_state == "not_attempted"


def test_identity_collect_non_current_search_runs_enriched_fallback_after_no_results():
    """Cycle-Audit-Fixes §2: identity_collect WITHOUT use_current_search and
    WITHOUT a fixed search_location runs the enriched plan. The bare-name
    query is still issued first; enriched variants follow when the kept set
    is empty after the primary."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 0
    browser.get_card_count.return_value = 0

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            workflow_mode="identity_collect",
            open_profile_on_likely_match=True,
        ),
        linkedin_brief=None,
    )
    # prepare_search WITHOUT a search_location → "auto" policy resolves to
    # enriched (no current-search hint, no fixed location).
    asyncio.run(resolver.prepare_search(""))

    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    issued_queries = [call.args[0] for call in browser.enter_search_string.await_args_list]
    assert len(issued_queries) >= 2, issued_queries
    # Bare quoted lookup name leads the plan.
    assert issued_queries[0] == '"Ada Lovelace"', issued_queries
    # And at least one enriched variant followed (carrying company/location/
    # title tokens from build_candidate_lookup_queries).
    assert any('"Ada Lovelace" AND' in q for q in issued_queries[1:]), issued_queries
    assert result.queries_tried == issued_queries


def test_fit_gated_save_runs_full_enriched_plan_with_bare_name_first():
    """Cycle-Audit-Fixes §1: fit_gated_save keeps the enriched plan from the
    first query (no name-first wrapping). The bare quoted lookup name still
    leads, then enriched variants follow without waiting for no-results."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 0
    browser.get_card_count.return_value = 0

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            workflow_mode="fit_gated_save",
            open_profile_on_likely_match=True,
        ),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search("New York"))

    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    issued_queries = [call.args[0] for call in browser.enter_search_string.await_args_list]
    # fit_gated_save runs the enriched plan in one go: bare name first, then
    # enriched fallbacks; the loop runs until plan exhausted because nothing
    # surfaced. (no_new_urls only fires once we have prior URLs to compare to.)
    assert len(issued_queries) >= 2, issued_queries
    assert issued_queries[0] == '"Ada Lovelace"', issued_queries
    # planned_queries records the full bounded plan; queries_tried records
    # the attempted prefix (here: the whole plan since none surfaced).
    assert result.planned_queries == issued_queries == result.queries_tried


@patch("linkedin.recruiter_identity_resolver.full_judge")
@patch("linkedin.recruiter_identity_resolver.extract_profile_from_innertext")
def test_provenance_queries_tried_is_attempted_prefix_not_full_plan(
    mock_extract,
    mock_judge,
):
    """Cycle-Audit-Fixes §4: queries_tried must be the attempted-only prefix.
    planned_queries carries the full bounded plan even when the loop short-
    circuits after the first query."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 1
    browser.get_card_snapshot.return_value = {
        "innertext": "\n".join(
            [
                "Eri B.",
                "Full-stack Developer at Plastic Labs",
                "New York · 3rd",
            ]
        ),
        "name": "Eri B.",
        "url": "/talent/profile/eri-b",
        "already_saved": False,
        "recruiter_activity": {"message_count": 0, "project_count": 0, "view_count": 0},
    }
    browser.get_profile_status_summary.return_value = {}
    mock_extract.return_value = CandidateProfileSummary(
        name="Eri Barrett",
        profile_url="/talent/profile/eri-b",
        headline="Full-stack Developer",
    )

    # fit_gated_save in non-current-search mode → planned_queries holds the
    # full enriched bounded plan (multiple entries). After the first surfaced
    # sole-init card the loop stops, so queries_tried is only that one query.
    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            workflow_mode="fit_gated_save",
            open_profile_on_likely_match=True,
            dry_run_save=True,
        ),
        linkedin_brief=MagicMock(),
    )
    asyncio.run(resolver.prepare_search(""))

    result = asyncio.run(resolver.resolve_lead(_eri_barrett_lead()))

    assert result.stop_reason == "single_surface_name_variant_stop"
    assert result.queries_tried == ['"Eri Barrett"']
    # planned_queries should contain MORE than the one attempted query
    # (because fit_gated_save plans the enriched fallbacks even though they
    # were never issued).
    assert len(result.planned_queries) > len(result.queries_tried)
    assert result.queries_tried[0] in result.planned_queries
    # Identity confirmation still ran (the post-loop sole-init branch fires
    # on the surfaced card without further mutation).
    assert result.identity_classification == "single_surface_name_variant_profile"
    mock_judge.assert_called()  # fit_gated_save calls full_judge after identity confirms


def test_query_expansion_policy_force_enriched_overrides_use_current_search():
    """Operator escape hatch: --query-expansion-policy enriched on the runner
    forces the enriched plan even in current-search mode (covers the rare
    case where the operator wants the legacy fallback chain)."""
    browser = AsyncMock()
    browser.get_card_slot_count.return_value = 0
    browser.get_card_count.return_value = 0

    resolver = RecruiterIdentityResolver(
        browser=browser,
        project_url="https://www.linkedin.com/talent/hire/123/search",
        config=RecruiterResolverConfig(
            workflow_mode="identity_collect",
            open_profile_on_likely_match=True,
            query_expansion_policy="enriched",
        ),
        linkedin_brief=None,
    )
    asyncio.run(resolver.use_existing_search("New York"))

    result = asyncio.run(resolver.resolve_lead(_make_lead()))

    issued_queries = [call.args[0] for call in browser.enter_search_string.await_args_list]
    assert len(issued_queries) >= 2, issued_queries
    # First query is still the bare quoted lookup name; enriched variants
    # follow because the operator explicitly opted in.
    assert issued_queries[0] == '"Ada Lovelace"', issued_queries
    assert any('"Ada Lovelace" AND' in q for q in issued_queries[1:]), issued_queries
    assert result.queries_tried == issued_queries
