"""Tests for OSS Maintainers Slice 6 — evidence + evaluation integration.

Two key contracts:

1. ``GitHubCandidate.to_evidence_text()`` is BYTE-IDENTICAL when the
   candidate has no ``maintainership`` payload (classic github
   briefs run unchanged per spec §11). When ``maintainership`` is
   set, a MAINTAINERSHIP EVIDENCE section appears.
2. ``assemble_github_full_evaluation_system(brief)`` is BYTE-
   IDENTICAL when ``brief.target_projects`` is empty. When it's
   set, a MAINTAINERSHIP-LEVEL EVALUATION block appears with the
   recruiter's project list and desired level.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone

import pytest

from github.judgment_templates import (
    GITHUB_FULL_EVALUATION_TEMPLATE,
    _DEFAULT_PORTFOLIO_YES_PATTERNS,
    _assemble_maintainership_block,
    assemble_github_full_evaluation_system,
)
from github.schemas import (
    ContactInfo,
    GitHubCandidate,
    GitHubRepo,
    GitHubSearchQuery,
    GitHubUser,
)


@pytest.fixture(autouse=True)
def _freeze_evidence_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Freeze evidence-text time for deterministic account age and repo span."""
    frozen = datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr("github.schemas._now_utc", lambda: frozen)


# ---------------------------------------------------------------------------
# Brief stub — pre-Slice-2 callers might construct Brief objects in tests
# without the new fields. Use a stub that exposes only the surface the
# evaluator template reads from.
# ---------------------------------------------------------------------------


class _BriefStub:
    """Minimal Brief-shaped stub for the evaluator template.

    Mirrors the methods :data:`GITHUB_FULL_EVALUATION_TEMPLATE` calls
    on a Brief; lets tests set ``target_projects`` /
    ``maintainership_level`` without standing up the loader pipeline.
    """

    def __init__(
        self,
        *,
        target_projects: list[str] | None = None,
        target_stacks: list[str] | None = None,
        maintainership_level: str = "contributor",
        role_level: str = "L7",
        minimum_years_experience: int = 7,
    ) -> None:
        self.role_title = "Staff infra engineer"
        self.role_level = role_level
        self.role_summary = "Infra leadership."
        self.minimum_years_experience = minimum_years_experience
        self.minimum_bar_description = "Has shipped infra at scale."
        self.target_projects = list(target_projects or [])
        self.target_stacks = list(target_stacks or [])
        self.maintainership_level = maintainership_level

    def capability_area_block(self) -> str:
        return "1. Infra"

    def depth_block(self) -> str:
        return "Builder = ships infra."

    def non_fit_block(self) -> str:
        return "(none)"

    def non_fit_override_rule_block(self) -> str:
        return "(none)"

    def employer_signal_block(self) -> str:
        return "(none)"

    def inferential_save_block(self) -> str:
        return "(none)"

    def discriminating_skills_examples(self) -> str:
        return "kubernetes, etcd"


# ---------------------------------------------------------------------------
# to_evidence_text() — byte-identical without maintainership
# ---------------------------------------------------------------------------


def _minimal_candidate() -> GitHubCandidate:
    return GitHubCandidate(
        user=GitHubUser(
            username="alice",
            name="Alice Doe",
            bio="ML engineer",
            company="ExampleCorp",
            followers=200,
            public_repos=10,
            created_at="2018-01-01T00:00:00Z",
            profile_url="https://github.com/alice",
        ),
    )


_MINIMAL_EVIDENCE_TEXT = (
    "Username: alice\n"
    "Name: Alice Doe\n"
    "Bio: ML engineer\n"
    "Company: ExampleCorp\n"
    "Location: (none)\n"
    "Followers: 200\n"
    "Public repos: 10\n"
    "Account created: 2018-01-01\n"
    "Account age: 8.0 years (as of 2026-01-01)\n"
    "Profile URL: https://github.com/alice\n"
    "\n"
    "═══ REPOSITORIES ═══"
)


def test_evidence_text_byte_identical_when_registry_evidence_none() -> None:
    """Absent registry_evidence renders byte-identically."""

    candidate = _minimal_candidate()
    candidate.registry_evidence = None
    assert candidate.to_evidence_text() == _MINIMAL_EVIDENCE_TEXT
    assert "REGISTRY MAINTAINERSHIP" not in candidate.to_evidence_text()


def test_evidence_text_byte_identical_when_project_quality_absent() -> None:
    """Absent portfolio_summary.project_quality renders byte-identically."""

    candidate = _minimal_candidate()
    candidate.portfolio_summary = {"toolchain_detected": {"frameworks": []}}
    assert "project_quality" not in candidate.portfolio_summary
    assert candidate.to_evidence_text() == _MINIMAL_EVIDENCE_TEXT
    assert "Project quality" not in candidate.to_evidence_text()


def test_evidence_text_includes_registry_maintainership_when_set() -> None:
    candidate = _minimal_candidate()
    candidate.registry_evidence = {
        "declared_roles": [
            {
                "hub": "npm",
                "handle": "sindresorhus",
                "package": "chalk",
                "role": "maintainer",
                "corroborated_github_login": "sindresorhus",
            },
        ],
        "packages": [
            {
                "hub": "npm",
                "name": "chalk",
                "downloads_last_month": 2_100_000,
                "downloads_window": "last-month",
                "reverse_dependencies": 340,
                "latest_release": "5.3.0",
                "release_cadence": "steady",
                "deprecated": False,
            },
        ],
    }
    text = candidate.to_evidence_text()

    assert "═══ REGISTRY MAINTAINERSHIP (declared) ═══" in text
    assert "Declared maintainer of chalk on npm" in text
    assert "2,100,000 downloads (last month)" in text
    assert "340 reverse dependencies" in text
    assert "Registry handle: sindresorhus" in text
    assert "Corroborated GitHub login: sindresorhus" in text
    assert "downloads/month" not in text


def test_evidence_text_registry_downloads_omit_window_when_unset() -> None:
    candidate = _minimal_candidate()
    candidate.registry_evidence = {
        "declared_roles": [
            {
                "hub": "crates",
                "handle": "dtolnay",
                "package": "serde",
                "role": "owner",
            },
        ],
        "packages": [
            {
                "hub": "crates",
                "name": "serde",
                "downloads_last_month": 500_000_000,
            },
        ],
    }
    text = candidate.to_evidence_text()

    assert "500,000,000 downloads" in text
    assert "last month" not in text
    assert "last 90 days" not in text
    assert "/month" not in text


def test_evidence_text_registry_downloads_use_last_90_days_window() -> None:
    candidate = _minimal_candidate()
    candidate.registry_evidence = {
        "declared_roles": [
            {
                "hub": "crates",
                "handle": "dtolnay",
                "package": "serde",
                "role": "owner",
            },
        ],
        "packages": [
            {
                "hub": "crates",
                "name": "serde",
                "downloads_last_month": 12_000_000,
                "downloads_window": "last-90-days",
            },
        ],
    }
    text = candidate.to_evidence_text()

    assert "12,000,000 downloads (last 90 days)" in text


def test_portfolio_text_byte_identical_when_registry_evidence_none() -> None:
    candidate = _minimal_candidate()
    candidate.registry_evidence = None
    baseline = candidate.to_portfolio_text()
    assert "Registry Maintainership" not in baseline

    candidate.portfolio_summary = {"ml_signal_strength": "low"}
    structured_baseline = candidate.to_portfolio_text()
    candidate.registry_evidence = None
    assert candidate.to_portfolio_text() == structured_baseline
    assert "Registry Maintainership" not in structured_baseline


def test_portfolio_text_includes_registry_maintainership_when_set() -> None:
    candidate = _minimal_candidate()
    candidate.portfolio_summary = {"ml_signal_strength": "medium"}
    candidate.registry_evidence = {
        "declared_roles": [
            {
                "hub": "npm",
                "package": "chalk",
                "role": "maintainer",
            },
        ],
        "packages": [
            {
                "hub": "npm",
                "name": "chalk",
                "downloads_last_month": 2_100_000,
                "downloads_window": "last-month",
                "reverse_dependencies": 340,
            },
        ],
    }
    text = candidate.to_portfolio_text()

    assert "Registry Maintainership (declared)" in text
    assert "Declared maintainer of chalk on npm" in text
    assert "2,100,000 downloads (last month)" in text
    assert "340 reverse dependencies" in text
    assert "Registry handle:" not in text


def test_evidence_text_includes_project_quality_line() -> None:
    candidate = _minimal_candidate()
    candidate.portfolio_summary = {
        "project_quality": {
            "repo": "lodash/lodash",
            "band": "high criticality",
            "score": 0.82,
        },
    }
    text = candidate.to_evidence_text()

    assert "═══ REPO ANALYSIS (from enrichment) ═══" in text
    assert "Project quality (lodash/lodash): high criticality (score 0.82)" in text


def test_full_eval_system_includes_declared_registry_tier_and_sparse_carveout() -> None:
    brief = _BriefStub(target_projects=[])
    prompt = assemble_github_full_evaluation_system(brief)

    assert "Tier 1.5" in prompt
    assert "DECLARED REGISTRY MAINTAINERSHIP OR GOVERNANCE ROSTER" in prompt
    assert "CODEOWNERS" in prompt
    assert "recipe-maintainers" in prompt
    assert (
        "Weigh either above stars/forks and above inferred contribution signals"
        in prompt
    )
    assert "REGISTRY SPARSE-PROFILE CARVE-OUT" in prompt
    assert "governance-roster maintainership" in prompt
    assert "is NOT sparse" in prompt


def test_full_eval_system_includes_tier_1_5_roster_wording() -> None:
    """Tier 1.5 extends declared registry weighting to governance rosters."""

    brief = _BriefStub(target_projects=[])
    prompt = assemble_github_full_evaluation_system(brief)

    assert "Tier 1.5 — HIGHEST (declared role):" in prompt
    assert "MAINTAINERS" in prompt
    assert "conda-forge recipe-maintainers" in prompt


def test_github_search_query_round_trips_new_fields() -> None:
    query = GitHubSearchQuery(
        id=7,
        name="npm maintainers",
        query="maintainers:chalk",
        channel="registry_maintainer_discovery",
        target_ecosystem="npmjs.org",
        target_packages=["chalk", "lodash"],
    )
    restored = GitHubSearchQuery.from_dict(query.to_dict())

    assert restored.channel == "registry_maintainer_discovery"
    assert restored.target_ecosystem == "npmjs.org"
    assert restored.target_packages == ["chalk", "lodash"]


def test_github_search_query_old_checkpoint_dict_loads_without_new_keys() -> None:
    legacy = {
        "id": 3,
        "name": "legacy",
        "query": "language:rust",
        "channel": "code_search",
        "status": "queued",
        "target_repo": "rust-lang/rust",
        "unknown_future_field": "ignored",
    }
    restored = GitHubSearchQuery.from_dict(legacy)

    assert restored.target_ecosystem == ""
    assert restored.target_packages == []
    assert restored.target_repo == "rust-lang/rust"


def test_evidence_text_byte_identical_when_maintainership_unset() -> None:
    """Spec §11: classic github briefs render byte-identically."""

    candidate = _minimal_candidate()
    candidate.maintainership = None
    text = candidate.to_evidence_text()
    assert "MAINTAINERSHIP EVIDENCE" not in text


def test_evidence_text_includes_maintainership_block_when_set() -> None:
    candidate = _minimal_candidate()
    candidate.maintainership = {
        "level": "maintainer",
        "confidence": 0.78,
        "role_certainty": "inferred",
        "evidence_sources": [
            "merge_authority:kubernetes/kubernetes:23PRs",
            "contributors_file:kubernetes/kubernetes",
        ],
        "signals": {"merge_authority": 1.5, "budget_exhausted": False},
    }
    text = candidate.to_evidence_text()

    assert "MAINTAINERSHIP EVIDENCE" in text
    assert "Role certainty: inferred (signal-based)" in text
    assert "maintainer" in text
    assert "0.78" in text
    assert "merge_authority:kubernetes/kubernetes:23PRs" in text


def test_maintainership_evidence_renders_certainty() -> None:
    candidate = _minimal_candidate()

    candidate.maintainership = {
        "level": "maintainer",
        "confidence": 0.78,
        "role_certainty": "inferred",
        "evidence_sources": ["merge_authority:kubernetes/kubernetes:23PRs"],
        "signals": {},
    }
    inferred_text = candidate.to_evidence_text()
    assert "Role certainty: inferred (signal-based)" in inferred_text
    assert "Classified level: maintainer (confidence 0.78)" in inferred_text

    candidate.maintainership = {
        "level": "maintainer",
        "confidence": 0.22,
        "role_certainty": "declared",
        "evidence_sources": [
            "declared:kubernetes/kubernetes:.github/CODEOWNERS",
            "merge_authority:kubernetes/kubernetes:2PRs",
        ],
        "corroboration": {"level": "contributor", "confidence": 0.22},
        "signals": {},
    }
    declared_text = candidate.to_evidence_text()
    assert "Declared level: maintainer (.github/CODEOWNERS in kubernetes/kubernetes)" in declared_text
    assert "Classifier corroborates at contributor (0.22)" in declared_text
    assert "Role certainty: DECLARED" not in declared_text
    assert "Classified level:" not in declared_text


def test_evidence_text_registry_declared_role_includes_source_file_when_set() -> None:
    candidate = _minimal_candidate()
    candidate.registry_evidence = {
        "declared_roles": [
            {
                "hub": "governance",
                "package": "kubernetes",
                "role": "code_owner",
                "source_file": ".github/CODEOWNERS",
            },
        ],
        "packages": [],
    }
    text = candidate.to_evidence_text()

    assert (
        "Declared code_owner of kubernetes on governance in .github/CODEOWNERS"
        in text
    )


def test_evidence_text_surfaces_budget_exhausted_note() -> None:
    candidate = _minimal_candidate()
    candidate.maintainership = {
        "level": "contributor",
        "confidence": 0.3,
        "evidence_sources": [],
        "signals": {"budget_exhausted": True},
    }
    text = candidate.to_evidence_text()

    assert "MAINTAINERSHIP EVIDENCE" in text
    assert "budget exhausted" in text


def test_evidence_text_omits_block_when_level_missing() -> None:
    """Defensive: malformed maintainership dict (no level) doesn't render."""

    candidate = _minimal_candidate()
    candidate.maintainership = {"confidence": 0.5}  # no level key
    text = candidate.to_evidence_text()

    assert "MAINTAINERSHIP EVIDENCE" not in text


def test_evidence_text_uses_domain_toolchain_and_target_project_headings() -> None:
    candidate = _minimal_candidate()
    candidate.portfolio_summary = {
        "toolchain_detected": {"frameworks": ["torch"]},
    }
    candidate.frontier_contributions = [
        {"repo": "kubernetes/kubernetes", "type": "maintainer", "detail": "active"},
    ]
    text = candidate.to_evidence_text()

    assert "═══ DOMAIN TOOLCHAIN DETECTED ═══" in text
    assert "═══ TARGET PROJECT CONTRIBUTIONS ═══" in text
    assert "FRONTIER TOOLCHAIN" not in text
    assert "FRONTIER REPO CONTRIBUTIONS" not in text


def test_portfolio_text_reads_legacy_frontier_contributions_key() -> None:
    candidate = _minimal_candidate()
    candidate.portfolio_summary = {
        "frontier_contributions": ["repo1: PR merged"],
    }
    text = candidate.to_portfolio_text()

    assert "Target Project Contributions" in text
    assert "Frontier Contributions" not in text


# ---------------------------------------------------------------------------
# assemble_github_full_evaluation_system — byte-identical contract
# ---------------------------------------------------------------------------


def test_full_eval_system_byte_identical_when_target_projects_empty() -> None:
    """Spec §11: classic github briefs render byte-identically.

    Compares the system prompt with `target_projects=[]` against the
    block returned by :func:`_assemble_maintainership_block` — the
    block must be empty so the template's `{maintainership_block}`
    slot renders as nothing extra (just two adjacent newlines around
    the slot).
    """

    brief = _BriefStub(target_projects=[])
    block = _assemble_maintainership_block(brief)
    assert block == ""

    prompt = assemble_github_full_evaluation_system(brief)
    assert "MAINTAINERSHIP-LEVEL EVALUATION" not in prompt
    assert "named target projects" not in prompt.lower()


def test_full_eval_system_includes_block_when_target_projects_set() -> None:
    brief = _BriefStub(
        target_projects=["kubernetes/kubernetes", "etcd-io/etcd"],
        maintainership_level="maintainer",
    )
    prompt = assemble_github_full_evaluation_system(brief)

    assert "MAINTAINERSHIP-LEVEL EVALUATION" in prompt
    assert "kubernetes/kubernetes" in prompt
    assert "etcd-io/etcd" in prompt
    assert "maintainer" in prompt


def test_full_eval_block_renders_each_recognized_level() -> None:
    """All three levels render cleanly into the block."""

    for level in ("contributor", "maintainer", "project_lead"):
        brief = _BriefStub(
            target_projects=["kubernetes/kubernetes"],
            maintainership_level=level,
        )
        block = _assemble_maintainership_block(brief)
        assert level in block, f"{level} missing from block"


def test_full_eval_block_describes_all_three_levels() -> None:
    """The block teaches the LLM what the three levels mean."""

    brief = _BriefStub(target_projects=["kubernetes/kubernetes"])
    block = _assemble_maintainership_block(brief)

    assert "contributor:" in block
    assert "maintainer:" in block
    assert "project_lead:" in block


def test_full_eval_block_contains_named_projects_in_visible_position() -> None:
    """Recruiter-named projects appear in the system prompt verbatim."""

    brief = _BriefStub(target_projects=["rust-lang/rust", "tokio-rs/tokio"])
    block = _assemble_maintainership_block(brief)

    assert "rust-lang/rust" in block
    assert "tokio-rs/tokio" in block


def test_github_eval_template_uses_neutral_project_contributions_vocab() -> None:
    """GitHub full-eval template must not hardcode a frontier-AI vertical."""

    assert "TARGET PROJECT CONTRIBUTIONS" in GITHUB_FULL_EVALUATION_TEMPLATE
    assert (
        "projects named in the brief (target_projects) and other recognized "
        "high-signal repositories in the candidate's domain"
        in GITHUB_FULL_EVALUATION_TEMPLATE
    )
    lowered = GITHUB_FULL_EVALUATION_TEMPLATE.lower()
    for term in (
        "frontier",
        "huggingface",
        "eleutherai",
        "vllm",
        "openrlhf",
        "axolotl",
        "swe-bench",
        "rlhf",
        "reward-model",
        "llm-evaluation",
        "machine learning",
        "artificial intelligence",
    ):
        assert term not in lowered, f"vertical literal still present: {term}"
    assert any(
        "target projects" in pattern.lower()
        for pattern in _DEFAULT_PORTFOLIO_YES_PATTERNS
    )


def test_evidence_text_includes_account_age_years() -> None:
    candidate = _minimal_candidate()
    candidate.user.created_at = "2018-01-01T00:00:00Z"
    text = candidate.to_evidence_text()

    lines = text.splitlines()
    created_idx = next(i for i, line in enumerate(lines) if line.startswith("Account created:"))
    age_idx = next(i for i, line in enumerate(lines) if line.startswith("Account age:"))
    profile_idx = next(i for i, line in enumerate(lines) if line.startswith("Profile URL:"))
    assert created_idx < age_idx < profile_idx
    assert lines[age_idx] == "Account age: 8.0 years (as of 2026-01-01)"

    candidate.user.created_at = ""
    text_without_age = candidate.to_evidence_text()
    assert "Account age:" not in text_without_age
    assert "Account created: unknown" in text_without_age


def test_evidence_text_includes_repo_activity_span() -> None:
    candidate = _minimal_candidate()
    candidate.top_repos = [
        GitHubRepo(
            name="alpha",
            created_at="2019-03-15T00:00:00Z",
            pushed_at="2025-11-20T00:00:00Z",
        ),
        GitHubRepo(
            name="beta",
            created_at="2021-06-10T00:00:00Z",
            pushed_at="2026-01-05T00:00:00Z",
        ),
    ]
    text = candidate.to_evidence_text()

    assert "═══ REPO ACTIVITY SPAN ═══" in text
    assert "First owned repo created: 2019-03" in text
    assert "Most recent push: 2026-01" in text
    assert "Repos pushed to in the last 12 months: 2 of 2" in text


def test_evidence_text_activity_span_ignores_forks() -> None:
    candidate = _minimal_candidate()
    candidate.top_repos = [
        GitHubRepo(
            name="real",
            created_at="2020-01-01T00:00:00Z",
            pushed_at="2025-06-01T00:00:00Z",
        ),
        GitHubRepo(
            name="ancient-fork",
            created_at="2010-01-01T00:00:00Z",
            pushed_at="2010-06-01T00:00:00Z",
            is_fork=True,
        ),
    ]
    text = candidate.to_evidence_text()
    span_section = text.split("═══ REPO ACTIVITY SPAN ═══")[1].split("═══")[0]

    assert "First owned repo created: 2020-01" in span_section
    assert "Most recent push: 2025-06" in span_section
    assert "2010-01" not in span_section
    assert "2010-06" not in span_section


def test_evidence_text_byte_identical_when_repo_dates_absent() -> None:
    candidate_no_repos = _minimal_candidate()
    candidate_blank_dates = _minimal_candidate()
    candidate_blank_dates.top_repos = [
        GitHubRepo(name="empty-dates", created_at="", pushed_at=""),
    ]

    assert "═══ REPO ACTIVITY SPAN ═══" not in candidate_no_repos.to_evidence_text()
    assert "═══ REPO ACTIVITY SPAN ═══" not in candidate_blank_dates.to_evidence_text()


def test_contribution_months_not_rendered() -> None:
    baseline = _minimal_candidate().to_evidence_text()

    candidate = _minimal_candidate()
    candidate.contribution_months = {
        "2019-03": 40,
        "2019-11": 200,
        "2020-06": 120,
    }
    assert candidate.to_evidence_text() == baseline
    assert "CONTRIBUTION HISTORY" not in candidate.to_evidence_text()


def test_full_eval_hierarchy_ranks_code_and_declared_over_described() -> None:
    brief = _BriefStub(target_projects=[], minimum_years_experience=4, role_level="")
    prompt = assemble_github_full_evaluation_system(brief)

    assert prompt.index("DOMAIN TOOLCHAIN USAGE") < prompt.index(
        "DECLARED REGISTRY MAINTAINERSHIP OR GOVERNANCE ROSTER"
    ) < prompt.index("BIO + PROFILE README SELF-DESCRIPTION")
    assert "CORROBORATOR ONLY" in prompt


def test_full_eval_no_hardcoded_inferential_shortcuts() -> None:
    brief = _BriefStub(target_projects=[], minimum_years_experience=4, role_level="")
    prompt = assemble_github_full_evaluation_system(brief)

    for forbidden in (
        ">200 followers",
        ">50 stars",
        "ML research/engineering",
        "GITHUB-SPECIFIC INFERENTIAL SAVE CONDITIONS",
    ):
        assert forbidden not in prompt


def test_full_eval_seniority_block_renders_for_principal() -> None:
    brief = _BriefStub(
        target_projects=[],
        role_level="principal",
        minimum_years_experience=8,
    )
    prompt = assemble_github_full_evaluation_system(brief)

    assert "SENIORITY EVIDENCE" in prompt
    assert "strong-junior" in prompt
    assert "ALREADY OPERATED" in prompt
    assert "grow into this role" not in prompt


def test_full_eval_seniority_block_absent_for_unspecified_level() -> None:
    brief = _BriefStub(
        target_projects=[],
        role_level="",
        minimum_years_experience=4,
    )
    prompt = assemble_github_full_evaluation_system(brief)

    assert "SENIORITY EVIDENCE" not in prompt
    assert (
        "Has done hands-on work with enough depth to grow into this role, even if "
        "their current domain is different."
    ) in prompt


def test_full_eval_seniority_triggers_on_years_alone() -> None:
    brief = _BriefStub(
        target_projects=[],
        role_level="",
        minimum_years_experience=8,
    )
    prompt = assemble_github_full_evaluation_system(brief)

    assert "SENIORITY EVIDENCE" in prompt


def test_seniority_trigger_word_boundaries_and_junior_veto() -> None:
    absent_cases = [
        _BriefStub(target_projects=[], role_level="junior team lead", minimum_years_experience=10),
        _BriefStub(target_projects=[], role_level="Mid-Senior", minimum_years_experience=10),
        _BriefStub(target_projects=[], role_level="Overhead ops", minimum_years_experience=4),
    ]
    for brief in absent_cases:
        prompt = assemble_github_full_evaluation_system(brief)
        assert "SENIORITY EVIDENCE" not in prompt

    present_cases = [
        _BriefStub(target_projects=[], role_level="Sr. Software Engineer", minimum_years_experience=4),
        _BriefStub(target_projects=[], role_level="principal", minimum_years_experience=4),
    ]
    for brief in present_cases:
        prompt = assemble_github_full_evaluation_system(brief)
        assert "SENIORITY EVIDENCE" in prompt


def test_full_eval_self_description_never_evidences_depth() -> None:
    brief = _BriefStub(target_projects=[], minimum_years_experience=4, role_level="")
    prompt = assemble_github_full_evaluation_system(brief)

    assert "can be evidenced by professional self-description" not in prompt
    assert "vague bio mention" not in prompt
