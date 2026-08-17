"""Tests for the ``_target_projects_drift`` cascade route (OSS Maintainers Slice 2).

Mirrors :mod:`tests.test_brief_polish` ``TestPath3Preservation`` and
``TestRoleTitlePreservation`` patterns. The drift helper is the load-
bearing contract: if the LLM polish drops, alters, or extends the
recruiter-named ``target_projects``, the cascade falls through to
the heuristic seed.

The set-equality + case-insensitive normalization contract is
exercised here so post-trial telemetry has a stable signal shape.
"""

from __future__ import annotations

from market_intelligence.brief_polish import (
    _normalize_target_projects,
    _target_projects_drift,
)


# ---------------------------------------------------------------------------
# _normalize_target_projects — coercion contract
# ---------------------------------------------------------------------------


def test_normalize_handles_non_list() -> None:
    assert _normalize_target_projects(None) == set()
    assert _normalize_target_projects("kubernetes/kubernetes") == set()
    assert _normalize_target_projects({"kubernetes/kubernetes"}) == set()


def test_normalize_drops_non_string_entries() -> None:
    assert _normalize_target_projects(["kubernetes/kubernetes", 42, None]) == {
        "kubernetes/kubernetes"
    }


def test_normalize_lowercases_and_strips() -> None:
    assert _normalize_target_projects(
        ["Kubernetes/Kubernetes", "  rust-lang/rust  ", ""]
    ) == {"kubernetes/kubernetes", "rust-lang/rust"}


def test_normalize_dedups() -> None:
    assert _normalize_target_projects(
        ["kubernetes/kubernetes", "kubernetes/kubernetes"]
    ) == {"kubernetes/kubernetes"}


# ---------------------------------------------------------------------------
# _target_projects_drift — passthrough cases (no drift)
# ---------------------------------------------------------------------------


def test_drift_none_when_seed_empty() -> None:
    """Classic-github briefs (no target_projects) pass cleanly."""

    seed = {"target_projects": []}
    polish = {"target_projects": []}
    assert _target_projects_drift(seed, polish) is None


def test_drift_none_when_seed_absent() -> None:
    """Missing key entirely is the same as empty list."""

    assert _target_projects_drift({}, {"target_projects": []}) is None


def test_drift_none_when_set_equal() -> None:
    seed = {"target_projects": ["kubernetes/kubernetes", "etcd-io/etcd"]}
    polish = {"target_projects": ["kubernetes/kubernetes", "etcd-io/etcd"]}
    assert _target_projects_drift(seed, polish) is None


def test_drift_none_when_order_differs() -> None:
    """Set-equality contract: order does not change the meaning."""

    seed = {"target_projects": ["kubernetes/kubernetes", "etcd-io/etcd"]}
    polish = {"target_projects": ["etcd-io/etcd", "kubernetes/kubernetes"]}
    assert _target_projects_drift(seed, polish) is None


def test_drift_none_when_case_differs() -> None:
    """Case-insensitive: GitHub treats owner/repo case-insensitively."""

    seed = {"target_projects": ["Kubernetes/Kubernetes"]}
    polish = {"target_projects": ["kubernetes/kubernetes"]}
    assert _target_projects_drift(seed, polish) is None


def test_drift_none_when_duplicates_normalize() -> None:
    seed = {"target_projects": ["kubernetes/kubernetes"]}
    polish = {"target_projects": ["kubernetes/kubernetes", "kubernetes/kubernetes"]}
    assert _target_projects_drift(seed, polish) is None


# ---------------------------------------------------------------------------
# _target_projects_drift — drift cases (cascade fires)
# ---------------------------------------------------------------------------


def test_drift_fires_when_polish_drops_set() -> None:
    seed = {"target_projects": ["kubernetes/kubernetes"]}
    polish = {"target_projects": []}
    descriptor = _target_projects_drift(seed, polish)
    assert descriptor is not None
    assert "dropped" in descriptor
    assert "kubernetes/kubernetes" in descriptor


def test_drift_fires_when_polish_omits_key() -> None:
    seed = {"target_projects": ["kubernetes/kubernetes"]}
    polish: dict = {}
    descriptor = _target_projects_drift(seed, polish)
    assert descriptor is not None
    assert "dropped" in descriptor


def test_drift_fires_when_polish_drops_a_member() -> None:
    seed = {"target_projects": ["kubernetes/kubernetes", "etcd-io/etcd"]}
    polish = {"target_projects": ["kubernetes/kubernetes"]}
    descriptor = _target_projects_drift(seed, polish)
    assert descriptor is not None
    assert "etcd-io/etcd" in descriptor
    assert "dropped=" in descriptor


def test_drift_fires_when_polish_adds_a_member() -> None:
    """LLM is not authorized to extend the target_projects set.

    A recruiter's named project list is the search anchor; adding a
    project the recruiter didn't name is a contract violation just
    as much as dropping one.
    """

    seed = {"target_projects": ["kubernetes/kubernetes"]}
    polish = {"target_projects": ["kubernetes/kubernetes", "rust-lang/rust"]}
    descriptor = _target_projects_drift(seed, polish)
    assert descriptor is not None
    assert "added=" in descriptor
    assert "rust-lang/rust" in descriptor


def test_drift_fires_when_polish_replaces_set() -> None:
    """Both add + drop reported when the LLM swaps projects."""

    seed = {"target_projects": ["kubernetes/kubernetes"]}
    polish = {"target_projects": ["rust-lang/rust"]}
    descriptor = _target_projects_drift(seed, polish)
    assert descriptor is not None
    assert "dropped=" in descriptor
    assert "added=" in descriptor


# ---------------------------------------------------------------------------
# Integration with the broader cascade — soft check that the helper
# doesn't disturb sibling-module preservation routes.
# ---------------------------------------------------------------------------


def test_drift_does_not_falsely_fire_on_design_rubric_only_briefs() -> None:
    """Designer briefs (no target_projects) sail through this check."""

    seed = {"design_rubric": {"principles": [{"name": "Hierarchy"}]}}
    polish = {"design_rubric": {"principles": [{"name": "Hierarchy"}]}}
    assert _target_projects_drift(seed, polish) is None
