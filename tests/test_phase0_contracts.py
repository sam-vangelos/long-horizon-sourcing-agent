"""Phase 0 characterization tests for frozen contracts."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import pytest

from github.schemas import GitHubProgress, GitHubSearchQuery
from shared.bias_controls import SAVE_DECISIONS as BIAS_SAVE_DECISIONS
from shared.brief_loader import Brief as NormalizedBrief, load_brief
from shared.brief_schema import Brief as V2Brief
from shared.contracts import (
    ACTIVE_FACIAL_DECISIONS,
    BRIEF_FAMILIES,
    COMPAT_FACIAL_DECISIONS,
    FACIAL_DECISIONS,
    FAILURE_DECISIONS,
    FULL_DECISIONS,
    GITHUB_QUERY_STATUSES,
    LINKEDIN_STRING_STATUSES,
    NON_SAVE_REVIEW_DECISIONS,
    NORMALIZED_BRIEF_REQUIRED_FIELDS,
    REVIEW_REASON_CODES,
    RUN_LOG_EVENTS,
    SAVE_DECISIONS,
    TARGET_CANDIDATE_LIFECYCLE,
    V2_BRIEF_REQUIRED_FIELDS,
)
import shared.judger as _judger
from shared.judger import is_failure_decision
from shared.runtime_state.store import (
    DEDUP_BLOCKING_DECISIONS,
    DEDUP_BLOCKING_LINKEDIN_DECISIONS,
    DEDUP_BLOCKING_RUNTIME_DECISIONS,
)
from shared.schemas import Progress, SearchString


ROOT = Path(__file__).parent.parent

_LEGACY_CONTRACT_BRIEF = ROOT / "config" / "FDL-Brazil" / "brief-brazil-real.json"
_V2_CONTRACT_BRIEF = ROOT / "config" / "Head-of-FDE" / "brief-head-fde-enterprise-ai-nyc-v5.json"


def test_brief_family_contract_is_frozen():
    assert BRIEF_FAMILIES == {"legacy", "v2"}


def test_normalized_brief_contract_fields_exist():
    actual_fields = {f.name for f in fields(NormalizedBrief)}
    assert NORMALIZED_BRIEF_REQUIRED_FIELDS <= actual_fields


def test_v2_brief_contract_fields_exist():
    actual_fields = {f.name for f in fields(V2Brief)}
    assert V2_BRIEF_REQUIRED_FIELDS <= actual_fields


@pytest.mark.skipif(
    not _LEGACY_CONTRACT_BRIEF.is_file() or not _V2_CONTRACT_BRIEF.is_file(),
    reason="Optional legacy Brazil / Head-of-FDE brief JSON not under config/",
)
def test_legacy_and_v2_briefs_load_under_current_contracts():
    legacy = load_brief(_LEGACY_CONTRACT_BRIEF)
    v2 = load_brief(_V2_CONTRACT_BRIEF)

    assert legacy.has_v2_schema is False
    assert v2.has_v2_schema is True

    for field_name in NORMALIZED_BRIEF_REQUIRED_FIELDS:
        assert hasattr(legacy, field_name)
        assert hasattr(v2, field_name)

    for field_name in V2_BRIEF_REQUIRED_FIELDS:
        assert hasattr(v2._new_brief, field_name)


def test_decision_contracts_align_with_current_helpers():
    for decision in FAILURE_DECISIONS:
        assert is_failure_decision(decision) is True

    for decision in ACTIVE_FACIAL_DECISIONS | COMPAT_FACIAL_DECISIONS | SAVE_DECISIONS | {"REJECT"}:
        assert is_failure_decision(decision) is False

    assert SAVE_DECISIONS == BIAS_SAVE_DECISIONS
    assert SAVE_DECISIONS <= FULL_DECISIONS


def test_current_status_defaults_fit_frozen_contracts():
    assert SearchString(id=1, name="x", boolean="y").status in LINKEDIN_STRING_STATUSES
    assert Progress(brief_name="test").strings == []

    assert GitHubSearchQuery(id=1, name="x", query="y", channel="user_search").status in GITHUB_QUERY_STATUSES
    assert GitHubProgress(brief_name="test").queries == []


def test_target_candidate_lifecycle_is_frozen_for_phase2():
    assert TARGET_CANDIDATE_LIFECYCLE == (
        "discovered",
        "snippet_extracted",
        "facial_started",
        "facial_terminal",
        "full_started",
        "full_terminal",
        "failed_retryable",
        "failed_terminal",
    )


def test_run_log_event_vocabulary_matches_current_emitters():
    discovered: set[str] = set()
    for source_root in (
        "cloris",
        "designer",
        "exec_search",
        "github",
        "linkedin",
        "market_intelligence",
        "researcher",
        "shared",
    ):
        for path in (ROOT / source_root).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Name) and func.id == "log_event":
                    if len(node.args) >= 2:
                        discovered.update(_literal_event_names(node.args[1]))
                elif (
                    path.name == "fallback_acquisition.py"
                    and isinstance(func, ast.Name)
                    and func.id == "record_event"
                ):
                    for keyword in node.keywords:
                        if keyword.arg == "event_type":
                            discovered.update(_literal_event_names(keyword.value))

    assert discovered == RUN_LOG_EVENTS


def _literal_event_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        return _literal_event_names(node.body) | _literal_event_names(node.orelse)
    return set()


# ---------------------------------------------------------------------------
# FACIAL_BORDERLINE -- Step A of the slice 12 promotion plan.
#
# At Step A the constant is a *type-system widening* only. The parser,
# validator, orchestrator, runtime-state store, projections, and persistence
# layer must NOT yet recognize FACIAL_BORDERLINE. The tests below pin the
# shape of that boundary so the type-system widening cannot accidentally
# leak into runtime behavior, and so a future drive-by promotion is forced
# to think first.
# ---------------------------------------------------------------------------


def test_facial_borderline_is_active_decision():
    assert "FACIAL_BORDERLINE" in ACTIVE_FACIAL_DECISIONS


def test_facial_borderline_is_not_compat_decision():
    assert "FACIAL_BORDERLINE" not in COMPAT_FACIAL_DECISIONS


def test_facial_borderline_is_facial_decision():
    assert "FACIAL_BORDERLINE" in FACIAL_DECISIONS


def test_facial_borderline_is_not_failure_decision():
    assert "FACIAL_BORDERLINE" not in FAILURE_DECISIONS
    assert is_failure_decision("FACIAL_BORDERLINE") is False


def test_facial_borderline_is_not_dedup_blocking():
    assert "FACIAL_BORDERLINE" not in DEDUP_BLOCKING_LINKEDIN_DECISIONS
    assert "FACIAL_BORDERLINE" not in DEDUP_BLOCKING_DECISIONS
    assert "FACIAL_BORDERLINE" not in DEDUP_BLOCKING_RUNTIME_DECISIONS


def test_facial_borderline_parallels_facial_yes_dedup_status():
    assert "FACIAL_YES" in ACTIVE_FACIAL_DECISIONS
    assert "FACIAL_YES" not in DEDUP_BLOCKING_LINKEDIN_DECISIONS
    assert "FACIAL_BORDERLINE" in ACTIVE_FACIAL_DECISIONS
    assert "FACIAL_BORDERLINE" not in DEDUP_BLOCKING_LINKEDIN_DECISIONS
    assert "FACIAL_NO" in ACTIVE_FACIAL_DECISIONS
    assert "FACIAL_NO" in DEDUP_BLOCKING_LINKEDIN_DECISIONS


def test_facial_borderline_is_valid_in_judger():
    """Step B widens ``_VALID_FACIAL`` to include ``FACIAL_BORDERLINE``.

    Step A's prior pin (``not in _VALID_FACIAL``) guarded the dark-constant
    invariant. Step B is the slice where that invariant flips: the parser
    and the validator gate both widen, while persistence stays binary
    because the orchestrator translates ``FACIAL_BORDERLINE`` to
    ``FACIAL_YES`` upstream of any persistence call.
    """
    assert "FACIAL_BORDERLINE" in _judger._VALID_FACIAL


# ---------------------------------------------------------------------------
# P4: bounded non-save review outcomes for ambiguous candidates.
#
# REVIEW_INFERRED and REVIEW_FLAGGED are full-stage terminal decisions that
# MUST NOT inflate save counts and MUST NOT trigger LinkedIn save side
# effects. The pins below guard the contract boundary so a drive-by edit
# can't quietly fold these into the SAVE family.
# ---------------------------------------------------------------------------


def test_review_decisions_are_in_full_decisions():
    assert "REVIEW_INFERRED" in FULL_DECISIONS
    assert "REVIEW_FLAGGED" in FULL_DECISIONS


def test_review_decisions_are_not_save_class():
    assert NON_SAVE_REVIEW_DECISIONS == frozenset(
        {"REVIEW_INFERRED", "REVIEW_FLAGGED"}
    )
    assert NON_SAVE_REVIEW_DECISIONS.isdisjoint(SAVE_DECISIONS)
    assert "REVIEW_INFERRED" not in SAVE_DECISIONS
    assert "REVIEW_FLAGGED" not in SAVE_DECISIONS


def test_review_decisions_are_not_failure_decisions():
    assert "REVIEW_INFERRED" not in FAILURE_DECISIONS
    assert "REVIEW_FLAGGED" not in FAILURE_DECISIONS
    assert is_failure_decision("REVIEW_INFERRED") is False
    assert is_failure_decision("REVIEW_FLAGGED") is False


def test_review_reason_codes_match_spec():
    assert REVIEW_REASON_CODES == frozenset(
        {
            "spot_check",
            "inferred_high_priority",
            "needs_more_evidence",
            "identity_unclear",
            "source_gap",
        }
    )


def test_review_decisions_are_valid_in_judger_full_validator():
    """The legacy JSON full-eval validator must accept REVIEW_* so the
    old-brief code path does not route a model-emitted review decision
    through ``parse_failure_decision``. The V2 structural path constructs
    OpusDecision directly from the parser output and does not consult
    this set; the legacy paths do.
    """
    assert "REVIEW_INFERRED" in _judger._VALID_FULL
    assert "REVIEW_FLAGGED" in _judger._VALID_FULL


def test_candidate_review_recorded_event_is_registered():
    assert "candidate_review_recorded" in RUN_LOG_EVENTS


# ---------------------------------------------------------------------------
# P6: recruiter recovery health states
# ---------------------------------------------------------------------------


def test_recruiter_health_states_are_importable():
    from linkedin.recruiter_recovery import RECRUITER_HEALTH_STATES
    assert "healthy" in RECRUITER_HEALTH_STATES
    assert "aw_snap" in RECRUITER_HEALTH_STATES
    assert "target_crashed" in RECRUITER_HEALTH_STATES
    assert len(RECRUITER_HEALTH_STATES) == 10
