import json

from github.reconciliation_input import (
    build_identity_resolution_experiment_cohort,
    export_identity_resolution_experiment_cohort,
    load_github_reconciliation_batch,
    load_saved_github_leads,
)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_saved_github_leads_prefers_profile_url_over_duplicate_display_name(tmp_path):
    candidates = [
        {
            "user": {
                "username": "ada-one",
                "name": "Ada Lovelace",
                "profile_url": "https://github.com/ada-one",
                "company": "Anthropic",
                "location": "New York",
            },
            "contact": {},
            "source_query": "q1",
            "source_strategy": "user_search",
        },
        {
            "user": {
                "username": "ada-two",
                "name": "Ada Lovelace",
                "profile_url": "https://github.com/ada-two",
                "company": "OpenAI",
                "location": "San Francisco",
            },
            "contact": {},
            "source_query": "q2",
            "source_strategy": "user_search",
        },
    ]
    judgments = [
        {
            "stage": "full",
            "decision": "SAVE",
            "candidate_name": "Ada Lovelace",
            "profile_url": "https://github.com/ada-two",
            "confidence": 0.9,
            "rationale": "Strong fit",
        }
    ]

    _write_jsonl(tmp_path / "candidates.jsonl", candidates)
    _write_jsonl(tmp_path / "final_judgments.jsonl", judgments)
    _write_jsonl(tmp_path / "outreach.jsonl", [])

    leads = load_saved_github_leads(tmp_path)

    assert len(leads) == 1
    assert leads[0].username == "ada-two"


def test_load_github_reconciliation_batch_normalizes_profile_urls(tmp_path):
    candidates = [
        {
            "user": {
                "username": "ada-two",
                "name": "Ada Lovelace",
                "profile_url": "https://github.com/ada-two/",
            },
            "contact": {},
        }
    ]
    judgments = [
        {
            "stage": "full",
            "decision": "SAVE",
            "candidate_name": "Ada Lovelace",
            "profile_url": "http://github.com/ada-two",
            "confidence": 0.9,
            "rationale": "Strong fit",
        }
    ]

    _write_jsonl(tmp_path / "candidates.jsonl", candidates)
    _write_jsonl(tmp_path / "final_judgments.jsonl", judgments)
    _write_jsonl(tmp_path / "outreach.jsonl", [])

    batch = load_github_reconciliation_batch(tmp_path)

    assert len(batch.leads) == 1
    assert batch.leads[0].username == "ada-two"


def test_load_saved_github_leads_skips_ambiguous_duplicate_names_without_profile_url(tmp_path):
    candidates = [
        {
            "user": {
                "username": "ada-one",
                "name": "Ada Lovelace",
                "profile_url": "https://github.com/ada-one",
            },
            "contact": {},
        },
        {
            "user": {
                "username": "ada-two",
                "name": "Ada Lovelace",
                "profile_url": "https://github.com/ada-two",
            },
            "contact": {},
        },
    ]
    judgments = [
        {
            "stage": "full",
            "decision": "SAVE",
            "candidate_name": "Ada Lovelace",
            "confidence": 0.9,
            "rationale": "Strong fit",
        }
    ]

    _write_jsonl(tmp_path / "candidates.jsonl", candidates)
    _write_jsonl(tmp_path / "final_judgments.jsonl", judgments)
    _write_jsonl(tmp_path / "outreach.jsonl", [])

    batch = load_github_reconciliation_batch(tmp_path)
    leads = batch.leads

    assert leads == []
    assert batch.stats.skipped_ambiguous_name == 1


def test_load_github_reconciliation_batch_dedups_repeated_username_by_name(tmp_path):
    """P6.1 re-appends the candidates.jsonl record for a username after
    classify() runs (pre-classify row, then a post-classify row with
    maintainership populated). Before the fix, usernames_by_name
    accumulated ["octocat", "octocat"] for that display name with no
    dedup, so the profile-url-less name-fallback path (requires exactly
    one matching username) wrongly treated a single candidate as an
    ambiguous duplicate and dropped the SAVE."""
    candidate_record = {
        "user": {
            "username": "octocat",
            "name": "The Octocat",
            "profile_url": "https://github.com/octocat",
        },
        "contact": {},
    }
    candidates = [
        candidate_record,
        dict(candidate_record, maintainership={"level": "maintainer"}),
    ]
    judgments = [
        {
            "stage": "full",
            "decision": "SAVE",
            "candidate_name": "The Octocat",
            "confidence": 0.9,
            "rationale": "Strong fit",
        }
    ]

    _write_jsonl(tmp_path / "candidates.jsonl", candidates)
    _write_jsonl(tmp_path / "final_judgments.jsonl", judgments)
    _write_jsonl(tmp_path / "outreach.jsonl", [])

    batch = load_github_reconciliation_batch(tmp_path)

    assert batch.stats.skipped_ambiguous_name == 0
    assert len(batch.leads) == 1
    assert batch.leads[0].username == "octocat"


def test_load_github_reconciliation_batch_counts_unmatched_profile_urls(tmp_path):
    candidates = [
        {
            "user": {
                "username": "ada-one",
                "name": "Ada Lovelace",
                "profile_url": "https://github.com/ada-one",
            },
            "contact": {},
        }
    ]
    judgments = [
        {
            "stage": "full",
            "decision": "SAVE",
            "candidate_name": "Ada Lovelace",
            "profile_url": "https://github.com/ada-missing",
            "confidence": 0.9,
            "rationale": "Strong fit",
        }
    ]

    _write_jsonl(tmp_path / "candidates.jsonl", candidates)
    _write_jsonl(tmp_path / "final_judgments.jsonl", judgments)
    _write_jsonl(tmp_path / "outreach.jsonl", [])

    batch = load_github_reconciliation_batch(tmp_path)

    assert batch.leads == []
    assert batch.stats.skipped_unmatched_profile_url == 1
    assert batch.stats.total_saved_judgments == 1


def test_build_identity_resolution_experiment_cohort_excludes_hints_from_primary(tmp_path):
    candidates = [
        {
            "user": {
                "username": "hinted",
                "name": "Hinted Person",
                "profile_url": "https://github.com/hinted",
                "company": "Anthropic",
                "location": "New York",
            },
            "contact": {"linkedin_url": "https://www.linkedin.com/in/hinted-person/"},
            "source_query": "q1",
            "source_strategy": "user_search",
        },
        {
            "user": {
                "username": "clean",
                "name": "Ada Lovelace",
                "profile_url": "https://github.com/clean",
                "company": "Anthropic",
                "location": "New York",
            },
            "contact": {},
            "source_query": "q2",
            "source_strategy": "user_search",
        },
    ]
    judgments = [
        {
            "stage": "full",
            "decision": "SAVE",
            "candidate_name": "Hinted Person",
            "profile_url": "https://github.com/hinted",
            "confidence": 0.95,
            "rationale": "Hinted fit",
        },
        {
            "stage": "full",
            "decision": "SAVE",
            "candidate_name": "Ada Lovelace",
            "profile_url": "https://github.com/clean",
            "confidence": 0.92,
            "rationale": "Clean fit",
        },
    ]

    _write_jsonl(tmp_path / "candidates.jsonl", candidates)
    _write_jsonl(tmp_path / "final_judgments.jsonl", judgments)
    _write_jsonl(tmp_path / "outreach.jsonl", [])

    cohort = build_identity_resolution_experiment_cohort(tmp_path, primary_bucket_size=1, sanity_size=1)

    assert [lead.github_username for lead in cohort["primary"]] == ["clean"]
    assert [lead.github_username for lead in cohort["sanity"]] == ["hinted"]


def test_export_identity_resolution_experiment_cohort_writes_template_artifacts(tmp_path):
    candidates = [
        {
            "user": {
                "username": "ada",
                "name": "Ada Lovelace",
                "profile_url": "https://github.com/ada",
                "company": "Anthropic",
                "location": "New York",
            },
            "contact": {},
            "source_query": "q1",
            "source_strategy": "user_search",
        }
    ]
    judgments = [
        {
            "stage": "full",
            "decision": "SAVE",
            "candidate_name": "Ada Lovelace",
            "profile_url": "https://github.com/ada",
            "confidence": 0.95,
            "rationale": "fit",
        }
    ]
    _write_jsonl(tmp_path / "candidates.jsonl", candidates)
    _write_jsonl(tmp_path / "final_judgments.jsonl", judgments)
    _write_jsonl(tmp_path / "outreach.jsonl", [])

    export_dir = tmp_path / "experiment"
    paths = export_identity_resolution_experiment_cohort(
        tmp_path,
        export_dir,
        primary_bucket_size=1,
        sanity_size=0,
    )

    for path in paths.values():
        assert path.exists()


def test_build_identity_resolution_experiment_cohort_falls_back_to_saves_when_final_judgments_empty(tmp_path):
    candidates = [
        {
            "user": {
                "username": "ada",
                "name": "Ada Lovelace",
                "profile_url": "https://github.com/ada",
                "company": "Anthropic",
                "location": "New York",
                "bio": "Research engineer",
            },
            "contact": {},
            "source_query": "q1",
            "source_strategy": "user_search",
        }
    ]
    saves = [
        {
            "username": "ada",
            "name": "Ada Lovelace",
            "github_url": "https://github.com/ada",
            "location": "New York",
            "bio": "Research engineer",
            "company": "Anthropic",
            "decision": "SAVE",
            "confidence": 0.9,
            "decision_path": "DIRECT:test",
        }
    ]

    _write_jsonl(tmp_path / "candidates.jsonl", candidates)
    _write_jsonl(tmp_path / "final_judgments.jsonl", [])
    _write_jsonl(tmp_path / "outreach.jsonl", [])
    _write_jsonl(tmp_path / "saves.jsonl", saves)

    cohort = build_identity_resolution_experiment_cohort(tmp_path, primary_bucket_size=1, sanity_size=0)

    assert len(cohort["primary"]) == 1
    assert cohort["primary"][0].github_username == "ada"
