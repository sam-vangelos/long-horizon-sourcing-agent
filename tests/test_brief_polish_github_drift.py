"""Brief-polish drift cascades for GitHub OSS Maintainers fields — audit Move #24.

Adds preservation contracts for the two GitHub fields the existing
``_target_projects_drift`` doesn't cover:

- ``target_stacks`` — recruiter-named language / framework tags
  (set-equality contract, sibling of target_projects)
- ``maintainership_level`` — the recruiter's classification floor
  (equality contract; once Move #9's
  ``lower_maintainership_threshold`` hunk lands, the polish step
  must not silently raise it back)

Mirrors :mod:`tests.test_brief_polish_target_projects` patterns —
passthrough cases, drift descriptors, and the cross-cascade contract
that the polish run falls through to the heuristic seed when drift
fires.
"""

from __future__ import annotations

from market_intelligence.brief_polish import (
    _maintainership_level_drift,
    _normalize_target_stacks,
    _target_stacks_drift,
)


# ---------------------------------------------------------------------------
# _normalize_target_stacks — coercion contract
# ---------------------------------------------------------------------------


def test_normalize_stacks_handles_non_list() -> None:
    assert _normalize_target_stacks(None) == set()
    assert _normalize_target_stacks("rust") == set()
    assert _normalize_target_stacks({"rust"}) == set()


def test_normalize_stacks_drops_non_string_entries() -> None:
    assert _normalize_target_stacks(["rust", 42, None, ""]) == {"rust"}


def test_normalize_stacks_lowercases_and_strips() -> None:
    assert _normalize_target_stacks(
        ["Rust", "  Distributed-Systems  ", "kubernetes"]
    ) == {"rust", "distributed-systems", "kubernetes"}


def test_normalize_stacks_dedups() -> None:
    assert _normalize_target_stacks(["rust", "rust", "Rust"]) == {"rust"}


# ---------------------------------------------------------------------------
# _target_stacks_drift — passthrough cases
# ---------------------------------------------------------------------------


def test_stacks_drift_none_when_seed_empty() -> None:
    assert _target_stacks_drift(seeded={}, polished={"target_stacks": []}) is None


def test_stacks_drift_none_when_set_equal() -> None:
    seed = {"target_stacks": ["rust", "kubernetes"]}
    polish = {"target_stacks": ["rust", "kubernetes"]}
    assert _target_stacks_drift(seeded=seed, polished=polish) is None


def test_stacks_drift_none_when_order_differs() -> None:
    seed = {"target_stacks": ["rust", "kubernetes"]}
    polish = {"target_stacks": ["kubernetes", "rust"]}
    assert _target_stacks_drift(seeded=seed, polished=polish) is None


# ---------------------------------------------------------------------------
# _target_stacks_drift — drift cases
# ---------------------------------------------------------------------------


def test_stacks_drift_when_polish_drops_field() -> None:
    seed = {"target_stacks": ["rust", "kubernetes"]}
    polish = {}
    drift = _target_stacks_drift(seeded=seed, polished=polish)
    assert drift is not None
    assert "dropped" in drift


def test_stacks_drift_when_polish_drops_some_stacks() -> None:
    seed = {"target_stacks": ["rust", "kubernetes", "etcd"]}
    polish = {"target_stacks": ["rust"]}
    drift = _target_stacks_drift(seeded=seed, polished=polish)
    assert drift is not None
    assert "dropped" in drift
    assert "etcd" in drift
    assert "kubernetes" in drift


def test_stacks_drift_when_polish_adds_stacks() -> None:
    seed = {"target_stacks": ["rust"]}
    polish = {"target_stacks": ["rust", "go"]}
    drift = _target_stacks_drift(seeded=seed, polished=polish)
    assert drift is not None
    assert "added" in drift
    assert "go" in drift


# ---------------------------------------------------------------------------
# _maintainership_level_drift — passthrough cases
# ---------------------------------------------------------------------------


def test_maintainership_drift_none_when_seed_absent() -> None:
    assert (
        _maintainership_level_drift(
            seeded={}, polished={"maintainership_level": "maintainer"}
        )
        is None
    )


def test_maintainership_drift_none_when_seed_unrecognized() -> None:
    """Unknown level strings (typos, garbage) don't trigger preservation."""

    seed = {"maintainership_level": "junior_contributor"}
    polish = {"maintainership_level": "maintainer"}
    assert _maintainership_level_drift(seeded=seed, polished=polish) is None


def test_maintainership_drift_none_when_equal_case_insensitive() -> None:
    seed = {"maintainership_level": "Maintainer"}
    polish = {"maintainership_level": "maintainer"}
    assert _maintainership_level_drift(seeded=seed, polished=polish) is None


# ---------------------------------------------------------------------------
# _maintainership_level_drift — drift cases (the move's load-bearing case)
# ---------------------------------------------------------------------------


def test_maintainership_drift_when_polish_drops_field() -> None:
    seed = {"maintainership_level": "contributor"}
    polish = {}
    drift = _maintainership_level_drift(seeded=seed, polished=polish)
    assert drift is not None
    assert "dropped" in drift
    assert "contributor" in drift


def test_maintainership_drift_when_polish_raises_floor_silently() -> None:
    """The Move #9 reflection composer proposes lowering the floor;
    once a recruiter has approved that hunk, the brief carries the
    lowered floor. The polish step MUST NOT silently raise it back."""

    seed = {"maintainership_level": "contributor"}
    polish = {"maintainership_level": "maintainer"}
    drift = _maintainership_level_drift(seeded=seed, polished=polish)
    assert drift is not None
    assert "contributor" in drift
    assert "maintainer" in drift


def test_maintainership_drift_when_polish_lowers_floor_silently() -> None:
    """The reverse: if the recruiter set project_lead and polish
    silently lowers it, that's also drift."""

    seed = {"maintainership_level": "project_lead"}
    polish = {"maintainership_level": "maintainer"}
    drift = _maintainership_level_drift(seeded=seed, polished=polish)
    assert drift is not None
    assert "project_lead" in drift
    assert "maintainer" in drift


# ---------------------------------------------------------------------------
# Round-trip: a polish-step output that preserves both fields passes
# the cascade cleanly (the canonical post-Move-9 happy path)
# ---------------------------------------------------------------------------


def test_polish_preserves_lowered_floor_after_recruiter_approved_hunk() -> None:
    """End-to-end happy path: recruiter approved a Move #9 hunk that
    lowered maintainership_level from `project_lead` to `maintainer`.
    The brief now carries the lowered value. A subsequent polish run
    that preserves the value passes the cascade with no drift."""

    seed = {
        "target_projects": ["kubernetes/kubernetes"],
        "target_stacks": ["rust", "kubernetes"],
        "maintainership_level": "maintainer",
    }
    polish = {
        "target_projects": ["kubernetes/kubernetes"],
        "target_stacks": ["rust", "kubernetes"],
        "maintainership_level": "maintainer",
    }
    assert _maintainership_level_drift(seeded=seed, polished=polish) is None
    assert _target_stacks_drift(seeded=seed, polished=polish) is None
