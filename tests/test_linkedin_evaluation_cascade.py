"""Tests for the evaluation cascade (D6, D7, D8).

D6: Shadow profile-probe cascade — probe runs, full eval still executes.
D7: Audit sampling and false-negative guard — metrics on probe/full-eval disagreement.
D8: Controlled cascade activation criteria — off-by-default, threshold-gated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from linkedin.evaluation_cascade import (
    AuditSampler,
    CascadeActivationPolicy,
    CascadeDecision,
    CascadePolicy,
    CascadeRecord,
    ProbeDecision,
    ProfileProbe,
)
from shared.brief_loader import load_brief

_REPO_ROOT = Path(__file__).resolve().parents[1]
_V2_BRIEF_PATH = _REPO_ROOT / "config" / "FDL-Colombia" / "brief-fdl-colombia-v4.json"


# ---------------------------------------------------------------------------
# D6: ProfileProbe and CascadeDecision
# ---------------------------------------------------------------------------


def test_probe_decision_enum_values():
    assert ProbeDecision.FULL_EVAL.value == "full_eval"
    assert ProbeDecision.REJECT_WITHOUT_OPUS.value == "reject_without_opus"
    assert ProbeDecision.REVIEW_INFERRED.value == "review_inferred"
    assert ProbeDecision.REVIEW_FLAGGED.value == "review_flagged"


def test_cascade_decision_to_dict():
    d = CascadeDecision(
        probe_decision=ProbeDecision.FULL_EVAL,
        confidence=0.9,
        rationale="Strong match",
        probe_model="gpt-4o-mini",
        shadow=True,
    )
    result = d.to_dict()
    assert result["probe_decision"] == "full_eval"
    assert result["confidence"] == 0.9
    assert result["shadow"] is True


def test_probe_rejects_on_non_fit_pattern():
    probe = ProfileProbe()
    profile = {
        "headline": "Sales leader",
        "experiences": [{"title": "Sales Manager", "company": "", "summary_bullets": []}],
        "education": [],
        "skills_snippet": [],
    }
    signals = {
        "capability_areas": [{"area": "machine learning"}],
        "non_fit_patterns": [{"label": "sales"}],
    }
    result = probe.evaluate(profile, signals)
    assert result.probe_decision == ProbeDecision.REJECT_WITHOUT_OPUS


def test_probe_escalates_on_capability_match():
    probe = ProfileProbe()
    profile = {
        "headline": "Machine learning infrastructure",
        "experiences": [
            {
                "title": "ML Engineer",
                "company": "",
                "summary_bullets": ["Deep learning systems"],
            }
        ],
        "education": [],
        "skills_snippet": [],
    }
    signals = {
        "capability_areas": [{"area": "machine learning"}, {"area": "infrastructure"}],
        "non_fit_patterns": [],
    }
    result = probe.evaluate(profile, signals)
    assert result.probe_decision == ProbeDecision.FULL_EVAL


def test_probe_flags_on_no_signal_match():
    probe = ProfileProbe()
    profile = {
        "headline": "Team lead",
        "experiences": [{"title": "Manager", "company": "", "summary_bullets": []}],
        "education": [],
        "skills_snippet": [],
    }
    signals = {
        "capability_areas": [{"area": "distributed systems"}],
        "non_fit_patterns": [],
    }
    result = probe.evaluate(profile, signals)
    assert result.probe_decision == ProbeDecision.REVIEW_FLAGGED


def test_probe_review_inferred_on_partial_match():
    probe = ProfileProbe()
    profile = {
        "headline": "Backend systems",
        "experiences": [{"title": "Engineer", "company": "", "summary_bullets": []}],
        "education": [],
        "skills_snippet": [],
    }
    signals = {
        "capability_areas": [
            {"area": "backend"},
            {"area": "distributed systems"},
            {"area": "machine learning"},
        ],
        "non_fit_patterns": [],
    }
    result = probe.evaluate(profile, signals)
    assert result.probe_decision == ProbeDecision.REVIEW_INFERRED


def test_probe_shadow_mode_default():
    policy = CascadePolicy()
    assert policy.is_shadow_mode() is True
    probe = ProfileProbe(policy)
    result = probe.evaluate(
        {"headline": "x", "experiences": [], "education": [], "skills_snippet": []},
        {},
    )
    assert result.shadow is True


def test_probe_non_fit_string_pattern():
    """Non-fit patterns can be plain strings too."""
    probe = ProfileProbe()
    profile = {"headline": "Recruiter", "experiences": [], "education": [], "skills_snippet": []}
    signals = {"capability_areas": [], "non_fit_patterns": ["recruiter"]}
    result = probe.evaluate(profile, signals)
    assert result.probe_decision == ProbeDecision.REJECT_WITHOUT_OPUS


def test_probe_non_fit_label_does_not_match_across_field_boundaries():
    probe = ProfileProbe()
    profile = {
        "headline": "Director of Product",
        "experiences": [
            {"title": "Management Consultant", "company": "", "summary_bullets": []}
        ],
        "education": [],
        "skills_snippet": [],
    }
    signals = {
        "non_fit_patterns": [{"label": "product management"}],
    }
    result = probe.evaluate(profile, signals)
    assert result.probe_decision != ProbeDecision.REJECT_WITHOUT_OPUS


def test_probe_non_fit_label_matches_within_single_title_field():
    probe = ProfileProbe()
    profile = {
        "headline": "Business professional",
        "experiences": [
            {"title": "Product Management Lead", "company": "", "summary_bullets": []}
        ],
        "education": [],
        "skills_snippet": [],
    }
    signals = {
        "capability_areas": [],
        "non_fit_patterns": [{"label": "product management"}],
    }
    result = probe.evaluate(profile, signals)
    assert result.probe_decision == ProbeDecision.REJECT_WITHOUT_OPUS


def test_shadow_record_carries_probe_rationale():
    probe = ProfileProbe()
    rationale = "Non-fit pattern match: sales"
    decision = CascadeDecision(
        probe_decision=ProbeDecision.REJECT_WITHOUT_OPUS,
        confidence=0.7,
        rationale=rationale,
    )
    record = probe.record_shadow_outcome(
        candidate_name="Alice",
        profile_url="https://linkedin.com/in/alice",
        lane_id="ml-infra",
        probe=decision,
        full_eval_decision="SAVE",
        full_eval_confidence=0.9,
    )
    assert record.to_dict()["probe_rationale"] == rationale


# ---------------------------------------------------------------------------
# FIX (P10 actuate #2): the probe must read the REAL CandidateProfileSummary
# .to_dict() shape (shared/schemas.py) — headline/experiences/education/
# skills_snippet — not the never-emitted current_title/summary keys. A
# capability signal living only in an experience title/bullet or education
# field was invisible to the probe pre-fix; it always fell through to
# REVIEW_FLAGGED regardless of the candidate's real background.
# ---------------------------------------------------------------------------


def test_probe_reads_real_profile_summary_shape_not_stale_keys():
    probe = ProfileProbe()
    # A profile whose ONLY capability signal lives in an experience title +
    # bullet — the actual shape CandidateProfileSummary.to_dict() emits.
    # headline deliberately carries no capability words.
    profile = {
        "name": "Jane Doe",
        "profile_url": "https://linkedin.com/in/janedoe",
        "headline": "Senior Engineer",
        "experiences": [
            {
                "title": "Machine Learning Infrastructure Lead",
                "company": "Acme AI",
                "summary_bullets": ["Built distributed training systems"],
            }
        ],
        "education": [],
        "skills_snippet": [],
    }
    signals = {
        "capability_areas": [{"area": "machine learning"}, {"area": "distributed training"}],
        "non_fit_patterns": [],
    }

    result = probe.evaluate(profile, signals)

    assert result.probe_decision == ProbeDecision.FULL_EVAL


def test_probe_reads_education_and_skills_snippet_fields():
    probe = ProfileProbe()
    profile = {
        "name": "Jane Doe",
        "profile_url": "https://linkedin.com/in/janedoe",
        "headline": "Recent Graduate",
        "experiences": [],
        "education": [{"degree": "PhD", "school": "MIT", "field": "Robotics"}],
        "skills_snippet": ["ROS", "computer vision"],
    }
    signals = {
        "capability_areas": [{"area": "robotics"}, {"area": "computer vision"}],
        "non_fit_patterns": [],
    }

    result = probe.evaluate(profile, signals)

    assert result.probe_decision == ProbeDecision.FULL_EVAL


def test_probe_non_fit_pattern_matches_experience_not_just_headline():
    probe = ProfileProbe()
    profile = {
        "name": "Jane Doe",
        "profile_url": "https://linkedin.com/in/janedoe",
        "headline": "Business professional",
        "experiences": [{"title": "Sales Manager", "company": "Acme", "summary_bullets": []}],
        "education": [],
        "skills_snippet": [],
    }
    signals = {"capability_areas": [], "non_fit_patterns": [{"label": "sales"}]}

    result = probe.evaluate(profile, signals)

    assert result.probe_decision == ProbeDecision.REJECT_WITHOUT_OPUS


def test_probe_non_fit_label_ignores_non_title_profile_text():
    probe = ProfileProbe()
    profile = {
        "headline": "Senior Platform Engineer",
        "experiences": [
            {
                "title": "Staff Engineer",
                "company": "Consulting Partners",
                "summary_bullets": [
                    "Consulting engagement building distributed systems"
                ],
            }
        ],
        "education": [
            {"degree": "Consulting certificate", "school": "State", "field": "CS"}
        ],
        "skills_snippet": ["consulting"],
    }
    signals = {
        "capability_areas": [{"area": "distributed systems"}],
        "non_fit_patterns": [{"label": "consulting"}],
    }

    result = probe.evaluate(profile, signals)

    assert result.probe_decision == ProbeDecision.FULL_EVAL


# ---------------------------------------------------------------------------
# FIX 1 SEAM: shadow probe must read V2 sourcing signals (capability_area
# names + non_fit labels) from the structured brief, not via getattr on the
# compat Brief (which carries neither attr → silently empty → every
# candidate collapses to REVIEW_FLAGGED).
#
# This test loads a real V2 brief, builds signals via the new
# Brief.sourcing_signals() accessor, and runs ProfileProbe.evaluate across a
# strong-match / clear-non-fit / partial-match profile. It asserts the probe
# returns DIFFERENT decisions — specifically NOT all REVIEW_FLAGGED.
#
# Fail-before/pass-after confirmed: against the pre-fix orchestrator
# (getattr on the compat brief) the equivalent signals come back empty and
# the probe returns REVIEW_FLAGGED for all three profiles (verified by
# stashing brief_loader.sourcing_signals and reproducing the getattr read).
# ---------------------------------------------------------------------------


def test_sourcing_signals_accessor_reads_v2_schema():
    """The accessor pulls capability names + non-fit labels off _new_brief."""
    assert _V2_BRIEF_PATH.exists(), f"V2 brief fixture missing: {_V2_BRIEF_PATH}"
    brief = load_brief(str(_V2_BRIEF_PATH))
    assert brief.has_v2_schema is True

    signals = brief.sourcing_signals()

    # The compat Brief carries NEITHER attribute — proving a getattr-based read
    # (the bug) would have yielded empties here.
    assert not hasattr(brief, "capability_areas")
    assert not hasattr(brief, "non_fit_patterns")

    assert signals["capability_areas"], "expected non-empty capability area names"
    assert signals["non_fit_patterns"], "expected non-empty non-fit pattern labels"
    # Sanity: names/labels round-trip from the structured brief.
    assert signals["capability_areas"] == [
        n.strip() for n in brief._new_brief.capability_area_names()
    ]
    assert signals["non_fit_patterns"] == [
        nf.label.strip() for nf in brief._new_brief.non_fit_patterns
    ]


def test_sourcing_signals_legacy_brief_is_clearly_empty():
    """Legacy (pre-V2) briefs have no schema → explicitly-empty signals."""
    from shared.brief_loader import normalize_brief

    legacy = normalize_brief({"name": "legacy-role", "role_title": "Legacy Role"})
    assert legacy.has_v2_schema is False
    assert legacy.sourcing_signals() == {
        "capability_areas": [],
        "non_fit_patterns": [],
    }


def test_probe_discriminates_across_profiles_with_v2_signals():
    """SEAM: probe yields DIFFERENT decisions across distinct profiles — not all REVIEW_FLAGGED.

    This is the regression guard for FIX 1. With the correct V2 signals the
    probe separates a strong capability match, a clear non-fit, and a partial
    match. Under the old getattr-on-compat-brief path the signals were empty
    and all three returned REVIEW_FLAGGED.
    """
    assert _V2_BRIEF_PATH.exists(), f"V2 brief fixture missing: {_V2_BRIEF_PATH}"
    brief = load_brief(str(_V2_BRIEF_PATH))
    signals = brief.sourcing_signals()

    capability_names = signals["capability_areas"]
    non_fit_labels = signals["non_fit_patterns"]
    assert len(capability_names) >= 3, "fixture should have several capability areas"
    assert non_fit_labels, "fixture should have at least one non-fit label"

    probe = ProfileProbe()

    # Strong match: profile fields contain every capability-area name verbatim
    # (headline + skills_snippet, the real CandidateProfileSummary.to_dict()
    # shape), pushing match_ratio >= 0.5 → FULL_EVAL.
    strong_profile = {
        "headline": capability_names[1],
        "experiences": [],
        "education": [],
        "skills_snippet": list(capability_names),
    }
    # Clear non-fit: profile text contains a non-fit label verbatim → REJECT_WITHOUT_OPUS.
    non_fit_profile = {
        "headline": non_fit_labels[0],
        "experiences": [{"title": "analyst", "company": "", "summary_bullets": []}],
        "education": [],
        "skills_snippet": [],
    }
    # Partial / ambiguous: exactly one capability-area name, no non-fit → REVIEW_INFERRED
    # (0 < match_ratio < 0.5 given the fixture has many areas).
    partial_profile = {
        "headline": capability_names[1],
        "experiences": [
            {"title": "engineer", "company": "", "summary_bullets": ["general engineering work"]}
        ],
        "education": [],
        "skills_snippet": [],
    }

    strong = probe.evaluate(strong_profile, signals).probe_decision
    non_fit = probe.evaluate(non_fit_profile, signals).probe_decision
    partial = probe.evaluate(partial_profile, signals).probe_decision

    decisions = [strong, non_fit, partial]

    # Core seam assertion: the probe is no longer collapsing everything to one
    # bucket — and specifically not to REVIEW_FLAGGED (the empty-signals outcome).
    assert not all(d == ProbeDecision.REVIEW_FLAGGED for d in decisions), (
        f"probe returned REVIEW_FLAGGED for every profile — signals likely empty: {decisions}"
    )
    # And the three profiles genuinely separate.
    assert len(set(decisions)) == 3, f"expected 3 distinct decisions, got {decisions}"
    assert strong == ProbeDecision.FULL_EVAL
    assert non_fit == ProbeDecision.REJECT_WITHOUT_OPUS
    assert partial == ProbeDecision.REVIEW_INFERRED


def test_shadow_record_tracks_agreement():
    probe = ProfileProbe()
    decision = CascadeDecision(
        probe_decision=ProbeDecision.REJECT_WITHOUT_OPUS,
        confidence=0.7,
    )
    record = probe.record_shadow_outcome(
        candidate_name="Alice",
        profile_url="https://linkedin.com/in/alice",
        lane_id="ml-infra",
        probe=decision,
        full_eval_decision="SAVE",
        full_eval_confidence=0.9,
    )
    assert record.agreement is False  # probe rejected, full eval saved
    assert record.probe_decision == "reject_without_opus"
    assert record.full_eval_decision == "SAVE"


def test_shadow_record_agreement_when_both_pass():
    probe = ProfileProbe()
    decision = CascadeDecision(probe_decision=ProbeDecision.FULL_EVAL, confidence=0.8)
    record = probe.record_shadow_outcome(
        candidate_name="Bob",
        profile_url="",
        lane_id="platform",
        probe=decision,
        full_eval_decision="REJECT",
        full_eval_confidence=0.6,
    )
    assert record.agreement is True  # probe escalated, full eval rejected — no false negative


def test_shadow_records_accumulated():
    probe = ProfileProbe()
    for i in range(3):
        d = CascadeDecision(probe_decision=ProbeDecision.FULL_EVAL, confidence=0.8)
        probe.record_shadow_outcome(f"c{i}", "", "lane", d, "REJECT", 0.5)
    assert len(probe.shadow_records) == 3


# ---------------------------------------------------------------------------
# D7: AuditSampler
# ---------------------------------------------------------------------------


def _make_record(
    probe_decision: str = "full_eval",
    full_eval_decision: str = "REJECT",
    lane_id: str = "default",
    agreement: bool = True,
) -> CascadeRecord:
    return CascadeRecord(
        candidate_name="test",
        profile_url="",
        lane_id=lane_id,
        probe_decision=probe_decision,
        full_eval_decision=full_eval_decision,
        probe_confidence=0.5,
        full_eval_confidence=0.5,
        agreement=agreement,
    )


def test_audit_sampler_empty():
    sampler = AuditSampler()
    m = sampler.metrics()
    assert m["total_candidates"] == 0
    assert m["false_negative_rate"] == 0.0
    assert m["golden_save_preservation_rate"] == 1.0


def test_audit_sampler_false_negative_detection():
    records = [
        _make_record(probe_decision="reject_without_opus", full_eval_decision="SAVE"),
        _make_record(probe_decision="reject_without_opus", full_eval_decision="REJECT"),
        _make_record(probe_decision="full_eval", full_eval_decision="SAVE"),
    ]
    sampler = AuditSampler(records)
    m = sampler.metrics()
    assert m["total_candidates"] == 3
    assert m["false_negative_count"] == 1
    assert m["false_negative_rate"] == round(1 / 3, 4)


def test_audit_sampler_golden_save_preservation():
    records = [
        _make_record(probe_decision="full_eval", full_eval_decision="SAVE"),
        _make_record(probe_decision="review_inferred", full_eval_decision="SAVE"),
        _make_record(probe_decision="reject_without_opus", full_eval_decision="SAVE"),
    ]
    sampler = AuditSampler(records)
    m = sampler.metrics()
    # 3 golden saves, 2 preserved (full_eval + review_inferred), 1 missed (reject)
    assert m["golden_save_preservation_rate"] == round(2 / 3, 4)


def test_audit_sampler_suppression_opportunity():
    records = [
        _make_record(probe_decision="reject_without_opus", full_eval_decision="REJECT"),
        _make_record(probe_decision="reject_without_opus", full_eval_decision="REJECT"),
        _make_record(probe_decision="full_eval", full_eval_decision="REJECT"),
    ]
    sampler = AuditSampler(records)
    m = sampler.metrics()
    assert m["full_eval_suppression_opportunity"] == 2
    assert m["suppression_opportunity_rate"] == round(2 / 3, 4)


def test_audit_sampler_per_lane():
    records = [
        _make_record(lane_id="a", probe_decision="full_eval", full_eval_decision="SAVE"),
        _make_record(lane_id="a", probe_decision="reject_without_opus", full_eval_decision="REJECT"),
        _make_record(lane_id="b", probe_decision="reject_without_opus", full_eval_decision="SAVE"),
    ]
    sampler = AuditSampler(records)
    per_lane = sampler.per_lane_metrics()
    assert "a" in per_lane
    assert "b" in per_lane
    assert per_lane["a"]["false_negative_count"] == 0
    assert per_lane["b"]["false_negative_count"] == 1


def test_audit_sampler_add_records():
    sampler = AuditSampler()
    sampler.add_records([_make_record(), _make_record()])
    assert sampler.metrics()["total_candidates"] == 2


# ---------------------------------------------------------------------------
# D8: CascadeActivationPolicy
# ---------------------------------------------------------------------------


def test_activation_default_off():
    policy = CascadeActivationPolicy()
    sampler = AuditSampler([_make_record() for _ in range(300)])
    ok, reasons = policy.can_activate(sampler)
    assert ok is False
    assert any("active=False" in r for r in reasons)


def test_activation_fails_insufficient_data():
    policy = CascadeActivationPolicy(active=True, min_shadow_candidates=200)
    sampler = AuditSampler([_make_record() for _ in range(50)])
    ok, reasons = policy.can_activate(sampler)
    assert ok is False
    assert any("Insufficient" in r for r in reasons)


def test_activation_fails_high_false_negative_rate():
    records = [
        _make_record(probe_decision="reject_without_opus", full_eval_decision="SAVE")
        for _ in range(10)
    ] + [
        _make_record(probe_decision="full_eval", full_eval_decision="REJECT")
        for _ in range(200)
    ]
    policy = CascadeActivationPolicy(active=True, max_false_negative_rate=0.02)
    sampler = AuditSampler(records)
    ok, reasons = policy.can_activate(sampler)
    assert ok is False
    assert any("False-negative" in r for r in reasons)


def test_activation_succeeds_when_metrics_pass():
    records = [
        _make_record(probe_decision="full_eval", full_eval_decision="REJECT")
        for _ in range(200)
    ] + [
        _make_record(probe_decision="full_eval", full_eval_decision="SAVE")
        for _ in range(10)
    ]
    policy = CascadeActivationPolicy(active=True)
    sampler = AuditSampler(records)
    ok, reasons = policy.can_activate(sampler)
    assert ok is True
    assert reasons == []


def test_rollback_restores_full_eval():
    policy = CascadeActivationPolicy(active=True)
    records = [_make_record() for _ in range(300)]
    sampler = AuditSampler(records)
    ok, _ = policy.can_activate(sampler)
    assert ok is True

    # Rollback
    policy.active = False
    ok, reasons = policy.can_activate(sampler)
    assert ok is False
    assert any("active=False" in r for r in reasons)
