"""Regression tests for GitHub light prescreen bio term matching."""

from __future__ import annotations

from unittest.mock import MagicMock

from github.schemas import ContactInfo, GitHubCandidate, GitHubUser

from tests.test_github_pipeline import _make_pipeline


class _CapabilityArea:
    def __init__(self, name: str, key_terms: list[str] = [], github_code_signals: list[str] = []):
        self.name = name
        self.key_terms = key_terms
        self.github_code_signals = github_code_signals


class _NewBrief:
    def __init__(self, capability_areas: list[_CapabilityArea]):
        self.capability_areas = capability_areas


def _candidate(bio: str, public_repos: int = 2) -> GitHubCandidate:
    return GitHubCandidate(
        user=GitHubUser(
            username="prescreen_user",
            name="Prescreen User",
            bio=bio,
            public_repos=public_repos,
            profile_url="https://github.com/prescreen_user",
        ),
        contact=ContactInfo(),
    )


def _pipeline_with_brief(capability_areas: list[_CapabilityArea] | None = None):
    pipeline = _make_pipeline()
    new_brief = _NewBrief(capability_areas or [])
    pipeline.brief_obj.has_v2_schema = True
    pipeline.brief_obj._new_brief = new_brief
    return pipeline


def test_prescreen_does_not_match_ai_substring_in_maintainer() -> None:
    pipeline = _pipeline_with_brief()
    result = pipeline._prescreen_light(
        _candidate("Kubernetes maintainer, available for consulting")
    )
    assert result == "pass"


def test_prescreen_passes_when_brief_derived_term_present() -> None:
    pipeline = _pipeline_with_brief(
        [_CapabilityArea("Kubernetes", key_terms=["orchestration"])]
    )
    result = pipeline._prescreen_light(
        _candidate("Kubernetes orchestration consultant", public_repos=2)
    )
    assert result == "pass"


def test_prescreen_skips_when_brief_terms_absent() -> None:
    pipeline = _pipeline_with_brief(
        [_CapabilityArea("Kubernetes", key_terms=["orchestration"])]
    )
    result = pipeline._prescreen_light(
        _candidate("Frontend developer building React apps", public_repos=2)
    )
    assert result == "hard_skip"


def test_prescreen_empty_term_brief_does_not_term_skip() -> None:
    pipeline = _pipeline_with_brief([])
    result = pipeline._prescreen_light(
        _candidate("Generic software engineer", public_repos=2)
    )
    assert result == "pass"
