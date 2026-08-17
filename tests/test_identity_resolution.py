from shared.identity_resolution import (
    build_candidate_lookup_queries,
    build_person_lookup_name,
    choose_best_match,
    classify_recruiter_activity_pressure,
    resolve_direct_linkedin_hint,
    score_linkedin_identity_match,
)
from shared.reconciliation_schemas import LinkedInIdentityHints, RecruiterActivitySnapshot


def test_build_candidate_lookup_queries_is_stable_and_bounded():
    hints = LinkedInIdentityHints(
        candidate_name="Ada Lovelace",
        company="JPMorgan Chase & Co.",
        location="New York City Metropolitan Area",
        title="Head of Applied AI Platform",
    )

    queries = build_candidate_lookup_queries(hints)

    assert queries
    assert queries == build_candidate_lookup_queries(hints)
    assert len(queries) <= 5
    assert queries[0].startswith('"Ada Lovelace" AND')


def test_resolve_direct_linkedin_hint_returns_high_confidence_match():
    hints = LinkedInIdentityHints(
        candidate_name="Ada Lovelace",
        linkedin_url_hint="https://www.linkedin.com/in/ada-lovelace/",
        company="Anthropic",
        title="Research Engineer",
    )

    match = resolve_direct_linkedin_hint(hints)

    assert match is not None
    assert match.match_method == "direct_linkedin_hint"
    assert match.match_confidence == 0.96
    assert "Direct LinkedIn URL hint present" in match.evidence


def test_score_linkedin_identity_match_does_not_credit_unrelated_cards_for_url_hint():
    hints = LinkedInIdentityHints(
        candidate_name="Ada Lovelace",
        linkedin_url_hint="https://www.linkedin.com/in/ada-lovelace/",
        company="Anthropic",
    )

    match = score_linkedin_identity_match(
        hints,
        matched_name="Grace Hopper",
        matched_company="Anthropic",
        matched_profile_url="/talent/profile/grace",
    )

    assert "Direct LinkedIn URL hint present" not in match.evidence
    assert match.match_confidence < 0.6


def test_score_linkedin_identity_match_uses_activity_and_buckets_confidence():
    hints = LinkedInIdentityHints(
        candidate_name="Ada Lovelace",
        company="JPMorgan Chase",
        location="New York",
        title="Head of AI Platform",
    )
    activity = RecruiterActivitySnapshot(message_count=7, project_count=3, view_count=2)

    match = score_linkedin_identity_match(
        hints,
        matched_name="Ada Lovelace",
        matched_company="JPMorgan Chase & Co.",
        matched_title="Head of AI Platform",
        matched_location="New York, New York, United States",
        matched_profile_url="/talent/profile/ada",
        recruiter_activity=activity,
    )

    assert match.match_confidence >= 0.85
    assert match.novelty_pressure == "high"


def test_choose_best_match_returns_manual_review_for_close_candidates():
    strong = score_linkedin_identity_match(
        LinkedInIdentityHints(candidate_name="Ada Lovelace", company="Anthropic"),
        matched_name="Ada Lovelace",
        matched_company="Anthropic",
        matched_profile_url="/talent/profile/ada-1",
    )
    close = score_linkedin_identity_match(
        LinkedInIdentityHints(candidate_name="Ada Lovelace", company="Anthropic"),
        matched_name="Ada Lovelace",
        matched_company="Anthropic",
        matched_profile_url="/talent/profile/ada-2",
    )

    classification, best = choose_best_match([strong, close])

    assert classification == "manual_review"
    assert best is not None


def test_choose_best_match_returns_none_for_low_confidence_pool():
    weak = score_linkedin_identity_match(
        LinkedInIdentityHints(candidate_name="Ada Lovelace"),
        matched_name="Grace Hopper",
        matched_company="Different Corp",
        matched_profile_url="/talent/profile/grace",
    )

    classification, best = choose_best_match([weak])

    assert classification == "no_confident_match"
    assert best is None


def test_build_person_lookup_name_preserves_multi_token_names():
    assert build_person_lookup_name("Ada Lovelace", "ada") == "Ada Lovelace"
    assert build_person_lookup_name("Eri Barrett", "erosika") == "Eri Barrett"
    assert build_person_lookup_name("Keunwoo Choi", "keunwoochoi") == "Keunwoo Choi"


def test_build_person_lookup_name_single_token_falls_back_to_username_derived_surname():
    # Canonical motivating case from the dry-run: github name scraped as "Michael"
    # and the username holds the real surname signal.
    assert build_person_lookup_name("Michael", "mldangelo") == "Michael Mldangelo"


def test_build_person_lookup_name_single_token_accepts_github_url_as_username():
    assert (
        build_person_lookup_name("Michael", "https://github.com/mldangelo")
        == "Michael Mldangelo"
    )


def test_build_person_lookup_name_single_token_uses_camelcase_boundary_when_present():
    # Explicit camelcase boundary: "mldAngelo" -> ["mld", "Angelo"]; both pass the
    # safety filter, so both are appended (longest first). No apostrophes invented.
    assert build_person_lookup_name("Michael", "mldAngelo") == "Michael Angelo Mld"


def test_build_person_lookup_name_single_token_splits_on_underscore_hyphen_digit():
    assert build_person_lookup_name("Michael", "mld_angelo") == "Michael Angelo Mld"
    assert build_person_lookup_name("Michael", "mld-angelo-123") == "Michael Angelo Mld"
    # Digit boundary: the leading "sam" token is dropped because it equals the
    # candidate's single token case-insensitively; only "vangelos" is kept.
    assert build_person_lookup_name("Sam", "sam123vangelos") == "Sam Vangelos"


def test_build_person_lookup_name_single_token_skips_unsafe_usernames():
    # Username equals the candidate's single token case-insensitively -> no change.
    assert build_person_lookup_name("Michael", "Michael") == "Michael"
    # Username too short (single char) -> no change.
    assert build_person_lookup_name("Michael", "m") == "Michael"
    # Empty username -> no change.
    assert build_person_lookup_name("Michael", "") == "Michael"


def test_build_person_lookup_name_handles_empty_candidate_name():
    # Empty candidate name still uses the username-derived raw path as before, and
    # does NOT activate the single-token fallback (which requires a candidate name).
    assert build_person_lookup_name("", "ada-Lovelace") == "ada Lovelace"
    assert build_person_lookup_name("", "") == ""


def test_build_person_lookup_name_fallback_does_not_invent_punctuation():
    # Explicit negative case called out in the plan: we must not emit "Michael D'Angelo"
    # because the apostrophe is not present in the source username.
    assert "'" not in build_person_lookup_name("Michael", "mldangelo")
    assert "." not in build_person_lookup_name("Michael", "mldangelo")


def test_classify_recruiter_activity_pressure_is_explainable():
    assert classify_recruiter_activity_pressure(None) == "low"
    assert classify_recruiter_activity_pressure(
        RecruiterActivitySnapshot(message_count=2, project_count=1, view_count=1)
    ) == "low"
    assert classify_recruiter_activity_pressure(
        RecruiterActivitySnapshot(message_count=4, project_count=1, view_count=1)
    ) == "medium"
    assert classify_recruiter_activity_pressure(
        RecruiterActivitySnapshot(message_count=7, project_count=3, view_count=3)
    ) == "high"
