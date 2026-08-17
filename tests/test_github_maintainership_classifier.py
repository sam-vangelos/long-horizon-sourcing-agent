"""Tests for :mod:`github.maintainership` (OSS Maintainers Slice 4).

Two test layers:

1. **Per-signal unit tests** (this file) — exercise the scoring,
   text-mine helpers, and aggregation in isolation with mocked
   :class:`github.client.GitHubClient`. Always run; ship green now.

2. **Calibration agreement gate** — runs the classifier against the
   hand-classified fixture at
   :file:`tests/fixtures/github_maintainership_ground_truth.json`
   and asserts the spec §13.1 thresholds. Currently SKIPS when the
   fixture is empty (Slice 4 Part A is the human hand-classification
   work). Once populated, this gate either passes (Slice 4 ships) or
   fails (classifier needs re-tuning before ship).

The split exists because the classifier code can be unit-tested
deterministically with mocks; the agreement gate fundamentally
requires labels grounded in human judgment of real GitHub profiles.
Honest split: ship the code; surface the gate as deferred-to-human.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from github import maintainership as m
from github.client import GitHubClient


FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "github_maintainership_ground_truth.json"
)


# ---------------------------------------------------------------------------
# Per-signal unit tests
# ---------------------------------------------------------------------------


class TestUsernameInText:
    def test_matches_at_mention(self) -> None:
        assert m._username_in_text("alice", "PR reviewed by @alice")

    def test_matches_bare(self) -> None:
        assert m._username_in_text("alice", "alice <alice@example.com>")

    def test_case_insensitive(self) -> None:
        assert m._username_in_text("ALICE", "@alice")

    def test_does_not_match_substring(self) -> None:
        """`alice` must not match within `aliceland`."""

        assert not m._username_in_text("alice", "aliceland is real")

    def test_empty_inputs(self) -> None:
        assert not m._username_in_text("", "alice")
        assert not m._username_in_text("alice", "")


class TestUsernameInLeadSection:
    def test_finds_in_maintainers_block(self) -> None:
        text = """
        # Project X

        Long description here.

        ## Maintainers

        - @alice (lead)
        - @bob
        """
        assert m._username_in_lead_section("alice", text)

    def test_does_not_find_in_unrelated_section(self) -> None:
        text = """
        # Changelog

        Version 1.0 contributed by @alice
        """
        # Heuristic accepts "authors" too, so this is borderline; the
        # test pins that "@alice" in pure changelog without any
        # lead/maintainer header does not trigger.
        assert not m._username_in_lead_section("alice", text)


class TestLevelDistance:
    def test_zero_when_equal(self) -> None:
        assert m.level_distance("maintainer", "maintainer") == 0

    def test_one_step(self) -> None:
        assert m.level_distance("contributor", "maintainer") == 1
        assert m.level_distance("project_lead", "maintainer") == 1

    def test_two_steps(self) -> None:
        assert m.level_distance("contributor", "project_lead") == 2
        assert m.level_distance("project_lead", "contributor") == 2

    def test_unknown_level(self) -> None:
        # Defensive: unknown level returns a large distance.
        assert m.level_distance("core_team", "maintainer") > 2


class TestEvaluateAgreement:
    def test_empty_input_returns_zeros(self) -> None:
        result = m.evaluate_agreement([])
        assert result["exact_rate"] == 0.0
        assert result["within_one_rate"] == 0.0
        assert result["n"] == 0

    def test_perfect_agreement(self) -> None:
        pairs = [("maintainer", "maintainer"), ("contributor", "contributor")]
        result = m.evaluate_agreement(pairs)
        assert result["exact_rate"] == 1.0
        assert result["within_one_rate"] == 1.0
        assert result["non_adjacent_rate"] == 0.0
        assert result["n"] == 2

    def test_off_by_one(self) -> None:
        pairs = [("contributor", "maintainer")]
        result = m.evaluate_agreement(pairs)
        assert result["exact_rate"] == 0.0
        assert result["within_one_rate"] == 1.0
        assert result["non_adjacent_rate"] == 0.0

    def test_non_adjacent(self) -> None:
        pairs = [("project_lead", "contributor")]
        result = m.evaluate_agreement(pairs)
        assert result["exact_rate"] == 0.0
        assert result["within_one_rate"] == 0.0
        assert result["non_adjacent_rate"] == 1.0


# ---------------------------------------------------------------------------
# classify() integration — empty target_projects skip
# ---------------------------------------------------------------------------


def test_classify_returns_none_when_target_projects_empty() -> None:
    """Spec §11 behavior-preserving: classic github briefs skip classification."""

    client = GitHubClient(token="dummy")
    result = asyncio.run(m.classify("alice", [], client))
    assert result is None


def test_classify_returns_none_when_target_projects_all_blank() -> None:
    client = GitHubClient(token="dummy")
    result = asyncio.run(m.classify("alice", ["", "  "], client))
    assert result is None


# ---------------------------------------------------------------------------
# classify() integration — small mocked end-to-end
# ---------------------------------------------------------------------------


def _make_mocked_client(
    *,
    contents_returns: dict[tuple[str, str], str | None],
    readme_returns: dict[str, str | None],
    releases_returns: dict[str, list[dict]],
    pulls_returns: dict[str, list[dict]],
    reviews_returns: dict[tuple[str, int], list[dict]],
) -> GitHubClient:
    """Construct a GitHubClient with all maintainer-signal endpoints mocked."""

    client = GitHubClient(token="dummy")

    async def _contents(owner_repo: str, path: str) -> str | None:
        return contents_returns.get((owner_repo, path))

    async def _readme(owner_repo: str) -> str | None:
        return readme_returns.get(owner_repo)

    async def _releases(owner_repo: str, max_results: int = 30) -> list[dict]:
        return releases_returns.get(owner_repo, [])

    async def _pulls(
        owner_repo: str,
        state: str = "closed",
        sort: str = "updated",
        max_results: int = 100,
    ) -> list[dict]:
        return pulls_returns.get(owner_repo, [])

    async def _reviews(owner_repo: str, pull_number: int) -> list[dict]:
        return reviews_returns.get((owner_repo, pull_number), [])

    client.get_repo_contents = _contents  # type: ignore[assignment]
    client.get_repo_readme = _readme  # type: ignore[assignment]
    client.list_repo_releases = _releases  # type: ignore[assignment]
    client.list_repo_pulls = _pulls  # type: ignore[assignment]
    client.get_pull_reviews = _reviews  # type: ignore[assignment]
    return client


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from github import maintainer_signal_cache as mcache

    monkeypatch.setattr(mcache, "CACHE_ROOT", tmp_path / "cache")


def test_classify_project_lead_via_governance_and_releases() -> None:
    """A high-confidence project lead: governance + maintainers + many releases."""

    client = _make_mocked_client(
        contents_returns={
            ("kubernetes/kubernetes", "GOVERNANCE.md"): "Project Lead: @alice",
            ("kubernetes/kubernetes", "MAINTAINERS.md"): "@alice",
            ("kubernetes/kubernetes", "CONTRIBUTORS.md"): "@alice\n@bob\n",
        },
        readme_returns={"kubernetes/kubernetes": "## Maintainers\n- @alice (BDFL)"},
        releases_returns={
            "kubernetes/kubernetes": [
                {"tag_name": f"v{i}", "author": {"login": "alice"}, "published_at": "2026-04-01T00:00:00Z"}
                for i in range(5)
            ],
        },
        pulls_returns={
            "kubernetes/kubernetes": [
                {"number": i, "merged_by": {"login": "alice"}, "merged_at": "2026-04-01T00:00:00Z"}
                for i in range(20)
            ],
        },
        reviews_returns={
            ("kubernetes/kubernetes", i): [{"user": {"login": "alice"}}] for i in range(20)
        },
    )

    result = asyncio.run(
        m.classify("alice", ["kubernetes/kubernetes"], client)
    )
    assert result is not None
    assert result.level == "project_lead"
    assert result.confidence > 0.5
    assert any("governance" in s for s in result.evidence_sources)
    assert any("merge_authority" in s for s in result.evidence_sources)


def test_classify_contributor_with_no_maintainership_signals() -> None:
    """Drive-by contributor: bio-only, no merge authority."""

    client = _make_mocked_client(
        contents_returns={
            ("kubernetes/kubernetes", "GOVERNANCE.md"): None,
            ("kubernetes/kubernetes", "MAINTAINERS.md"): None,
            ("kubernetes/kubernetes", "CONTRIBUTORS.md"): None,
        },
        readme_returns={"kubernetes/kubernetes": "Generic README"},
        releases_returns={
            "kubernetes/kubernetes": [
                {"tag_name": "v1", "author": {"login": "someone-else"}, "published_at": "2026-04-01T00:00:00Z"}
            ],
        },
        pulls_returns={
            "kubernetes/kubernetes": [
                {"number": 1, "merged_by": {"login": "someone-else"}, "merged_at": "2026-04-01T00:00:00Z"}
            ],
        },
        reviews_returns={("kubernetes/kubernetes", 1): []},
    )

    result = asyncio.run(
        m.classify("alice", ["kubernetes/kubernetes"], client)
    )
    assert result is not None
    assert result.level == "contributor"
    assert result.confidence < 0.4
    assert result.evidence_sources == [] or all(
        "merge_authority" not in s for s in result.evidence_sources
    )


def test_classify_respects_api_budget() -> None:
    """When budget runs out, classification still returns with budget_exhausted=True."""

    client = _make_mocked_client(
        contents_returns={},
        readme_returns={},
        releases_returns={},
        pulls_returns={},
        reviews_returns={},
    )

    result = asyncio.run(
        m.classify(
            "alice",
            ["kubernetes/kubernetes", "etcd-io/etcd"],
            client,
            api_budget=0,
        )
    )
    assert result is not None
    assert result.signals.get("budget_exhausted") is True


# ---------------------------------------------------------------------------
# merge_declared_maintainership — declared roster precedence
# ---------------------------------------------------------------------------


def test_declared_roster_leads_and_classifier_corroborates() -> None:
    inferred = m.MaintainershipClassification(
        level="contributor",
        confidence=0.22,
        evidence_sources=["merge_authority:kubernetes/kubernetes:2PRs"],
        signals={"merge_authority": 0.5},
    )
    declared_entries = [
        {
            "role": "code_owner",
            "source_file": ".github/CODEOWNERS",
            "repo": "kubernetes/kubernetes",
            "hub": "governance",
        },
    ]

    merged = m.merge_declared_maintainership(declared_entries, inferred)

    assert merged["level"] == "maintainer"
    assert merged["role_certainty"] == "declared"
    assert merged["evidence_sources"][0] == (
        "declared:kubernetes/kubernetes:.github/CODEOWNERS"
    )
    assert "merge_authority:kubernetes/kubernetes:2PRs" in merged["evidence_sources"]
    assert merged["corroboration"] == {
        "level": "contributor",
        "confidence": 0.22,
    }


def test_no_declared_entries_keeps_inferred_unchanged() -> None:
    inferred = m.MaintainershipClassification(
        level="maintainer",
        confidence=0.78,
        evidence_sources=["maintainers_file:kubernetes/kubernetes"],
        signals={"maintainers_file": 1.0},
    )

    merged = m.merge_declared_maintainership([], inferred)

    assert merged == inferred.to_dict()
    assert merged["role_certainty"] == "inferred"
    assert "corroboration" not in merged


def test_declared_registry_entry_without_repo_uses_package_token() -> None:
    declared_entries = [
        {
            "hub": "crates",
            "package": "serde",
            "role": "owner",
        },
    ]

    merged = m.merge_declared_maintainership(declared_entries, None)

    assert merged["role_certainty"] == "declared"
    assert merged["evidence_sources"] == ["declared:serde:"]
    assert "declared::" not in merged["evidence_sources"]
    assert merged.get("confidence") is None
    assert "corroboration" not in merged


def test_no_declared_token_is_malformed() -> None:
    declared_entries = [
        {
            "hub": "crates",
            "handle": "dtolnay",
            "package": "serde",
            "role": "owner",
            "corroborated_github_login": "dtolnay",
        },
    ]

    merged = m.merge_declared_maintainership(declared_entries, None)

    for token in merged["evidence_sources"]:
        assert not token.endswith("::")
        assert token != "declared::"


def test_classification_to_dict_includes_role_certainty_inferred() -> None:
    classification = m.MaintainershipClassification(
        level="contributor",
        confidence=0.1,
    )

    assert classification.to_dict()["role_certainty"] == "inferred"


# ---------------------------------------------------------------------------
# Calibration fixture loader + agreement gate
# ---------------------------------------------------------------------------


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


def test_fixture_file_exists_and_is_valid_json() -> None:
    fixture = _load_fixture()
    assert "_meta" in fixture
    assert "entries" in fixture
    assert isinstance(fixture["entries"], list)


def test_fixture_meta_documents_ship_gate_thresholds() -> None:
    """The thresholds live in the fixture itself for reviewability."""

    meta = _load_fixture()["_meta"]
    gate = meta.get("ship_gate_thresholds")
    assert gate is not None
    assert gate["exact_level_match"] == 0.80
    assert gate["within_one_level_match"] == 0.95
    assert gate["non_adjacent_confusion_max"] == 0.02


@pytest.mark.skipif(
    not _load_fixture()["entries"],
    reason=(
        "Slice 4 calibration fixture is empty. Slice 4 ships in two parts: "
        "Part A (this fixture) is human hand-classification work; Part B "
        "(github/maintainership.py classifier) is shipped. Once the fixture "
        "carries 20-30 hand-classified entries per spec §13.1, this gate "
        "either passes (Slice 4 fully ships) or fails (classifier needs "
        "re-tuning before ship)."
    ),
)
def test_classifier_clears_agreement_gate_against_fixture() -> None:
    """Ship gate per spec §13.1.

    NOTE: this test is the load-bearing gate for Slice 4. It does not
    run today because the fixture is empty (see the skip reason). Once
    populated, asserting against this gate requires REAL GitHub API
    access to score each entry — which means the test will need either
    (a) a recorded HTTP cassette for each fixture entry, or (b) a
    network-marker so it only runs in calibration-mode CI. The minimum
    viable path is (a): record cassettes for each entry once, then the
    gate runs deterministically on every test execution.

    This stub asserts the structure; the harness for cassette-based
    execution is the natural extension.
    """

    fixture = _load_fixture()
    entries = fixture["entries"]
    assert entries, "Fixture is empty — should have skipped above."

    # When entries are present + cassettes are recorded, the body
    # below executes the classifier per entry and aggregates.
    # Until then, this assertion is the placeholder contract.
    pairs: list[tuple[str, str]] = []
    for entry in entries:
        # Real implementation will run `m.classify(...)` against a
        # cassette-backed client. Until cassettes ship, we read the
        # `predicted_level` field if the recorder pre-populated it,
        # else skip the entry. Slice 4 Part A's deliverable is
        # cassettes paired with hand-labels.
        predicted = entry.get("predicted_level")
        truth = entry.get("ground_truth_level")
        if predicted is None or truth is None:
            continue
        pairs.append((predicted, truth))

    if not pairs:
        pytest.skip(
            "Fixture has entries but no `predicted_level` populated. "
            "Re-record cassettes and run the calibration harness to "
            "populate predictions, then re-run this gate."
        )

    metrics = m.evaluate_agreement(pairs)
    gate = fixture["_meta"]["ship_gate_thresholds"]
    assert metrics["exact_rate"] >= gate["exact_level_match"], (
        f"Exact-level agreement {metrics['exact_rate']:.3f} below gate "
        f"{gate['exact_level_match']}. Re-tune SIGNAL_WEIGHTS or "
        f"LEVEL_THRESHOLDS in github/maintainership.py."
    )
    assert metrics["within_one_rate"] >= gate["within_one_level_match"], (
        f"Within-one-level agreement {metrics['within_one_rate']:.3f} "
        f"below gate {gate['within_one_level_match']}."
    )
    assert metrics["non_adjacent_rate"] <= gate["non_adjacent_confusion_max"], (
        f"Non-adjacent confusion {metrics['non_adjacent_rate']:.3f} above "
        f"ceiling {gate['non_adjacent_confusion_max']}."
    )
