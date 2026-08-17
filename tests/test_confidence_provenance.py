"""P6 (Wave 2 slice 10): measurements are never edited; fabricated values are
never silent (plans/sourcing-rigor-hardening.md, audit R6-F3/R6-F4 +
provenance).

- Confidence parse failure -> None + confidence_parse_failed flag, never a
  fabricated mid-scale 0.5.
- The evidence-density ±0.05 adjustment is DELETED: the recorded confidence
  is the model's stated value verbatim (the adjustment contaminated the GLM
  shadow-judge agreement data with values the judge never stated).
- Loader-filled band defaults carry provenance (band_source) so a 0.25–0.55
  band is attributable: authored vs template ride-along.
- Hypothesis confidence is a volume-aware Wilson lower bound, not a flat 0.7
  ("2 saves of 400" and "2 saves of 6" no longer validate identically).
"""

from __future__ import annotations

import pytest

from linkedin.judgment_templates import parse_full_evaluation_response


def _full_eval(body: str) -> str:
    return (
        "STEP_1_MATCH: ADJACENT\n"
        "STEP_1_AREA: Network optimization\n"
        "STEP_1_EVIDENCE: Led the distribution network redesign end to end across three regions.\n"
        "STEP_2_DEPTH: BUILDER\n"
        "STEP_2_EVIDENCE: Built the load-balancing model that set the network plan for two years.\n"
        "STEP_3_TRANSFERABILITY: TRANSFERABLE\n"
        "STEP_3_EVIDENCE: The optimization methodology transfers cleanly to the target context.\n"
        "CASE_FOR: Owned the design tradeoffs personally.\n"
        "CASE_AGAINST: Vertical differs from the target.\n"
        + body
    )


# ---------------------------------------------------------------------------
# 1. Parse failure -> None + flag, never a fabricated 0.5
# ---------------------------------------------------------------------------


def test_unparsable_confidence_returns_none_with_flag():
    result = parse_full_evaluation_response(
        _full_eval(
            "DECISION: SAVE\n"
            "CONFIDENCE: 0.72 (moderate)\n"
            "POST_SAVE_MODIFIER: NONE\n"
            "SUMMARY: Strong adjacent builder.\n"
        )
    )
    assert result.decision == "SAVE"
    assert result.confidence is None
    assert result.confidence_parse_failed is True


def test_missing_confidence_line_returns_none_with_flag():
    result = parse_full_evaluation_response(
        _full_eval(
            "DECISION: REJECT\n"
            "POST_SAVE_MODIFIER: NONE\n"
            "SUMMARY: Wrong depth.\n"
        )
    )
    assert result.confidence is None
    assert result.confidence_parse_failed is True


def test_structural_parse_failure_carries_none_confidence():
    result = parse_full_evaluation_response("")
    assert result.decision == "PARSE_FAILURE"
    assert result.confidence is None
    assert result.confidence_parse_failed is True


# ---------------------------------------------------------------------------
# 2. The stated value is recorded VERBATIM — the density adjustment is gone
# ---------------------------------------------------------------------------


def test_dense_evidence_save_keeps_stated_confidence_verbatim():
    """Old code bumped an ADJACENT save with 3 dense evidence fields from
    0.55 to 0.60. The measurement is no longer edited."""
    result = parse_full_evaluation_response(
        _full_eval(
            "DECISION: SAVE\n"
            "CONFIDENCE: 0.55\n"
            "POST_SAVE_MODIFIER: NONE\n"
            "SUMMARY: Strong adjacent builder.\n"
        )
    )
    assert result.confidence == pytest.approx(0.55)
    assert result.confidence_parse_failed is False


def test_sparse_evidence_save_keeps_stated_confidence_verbatim():
    """Old code dropped a sparse-evidence save from 0.62 to 0.57."""
    raw = (
        "STEP_1_MATCH: ADJACENT\n"
        "STEP_1_AREA: Network optimization\n"
        "STEP_1_EVIDENCE: N/A\n"
        "STEP_2_DEPTH: BUILDER\n"
        "STEP_2_EVIDENCE: short\n"
        "STEP_3_TRANSFERABILITY: TRANSFERABLE\n"
        "STEP_3_EVIDENCE: ok\n"
        "CASE_FOR: Thin.\n"
        "CASE_AGAINST: Thin.\n"
        "DECISION: SAVE\n"
        "CONFIDENCE: 0.62\n"
        "POST_SAVE_MODIFIER: NONE\n"
        "SUMMARY: Sparse but plausible.\n"
    )
    result = parse_full_evaluation_response(raw)
    assert result.confidence == pytest.approx(0.62)


# ---------------------------------------------------------------------------
# 3. Band provenance: loader-filled defaults are attributable
# ---------------------------------------------------------------------------


def _v2_payload(facial_calibration: dict) -> dict:
    return {
        "role_title": "Director of Supply Chain Operations",
        "role_level": "Director",
        "role_summary": "Owns network design.",
        "geography": "Chicago",
        "minimum_years_experience": 8,
        "minimum_bar_description": "Owned network design.",
        "linkedin_project": "test-project",
        "capability_areas": [
            {
                "name": "Network optimization",
                "description": "Designs networks.",
                "builder_signals": ["designed the network"],
                "user_signals": ["ran reports"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Designs the network.",
            "user_definition": "Operates the network.",
            "edge_case_guidance": "Ownership decides.",
        },
        "non_fit_patterns": [
            {
                "label": "Warehouse ops",
                "description": "Runs shifts.",
                "why_not": "No design ownership.",
                "examples": ["shift lead"],
            }
        ],
        "employer_signal_rules": [],
        "facial_calibration": facial_calibration,
    }


def test_loader_default_band_is_stamped_loader_default():
    from shared.brief_loader import _load_v2_brief

    brief = _load_v2_brief(
        _v2_payload({"fast_exit_patterns": ["career entirely in retail ops"]})
    )
    calibration = brief._new_brief.facial_calibration
    assert calibration.expected_yes_rate_low == pytest.approx(0.25)
    assert calibration.band_source == "loader_default"


def test_authored_band_is_stamped_preflight():
    from shared.brief_loader import _load_v2_brief

    brief = _load_v2_brief(
        _v2_payload(
            {
                "expected_yes_rate_low": 0.08,
                "expected_yes_rate_high": 0.22,
                "fast_exit_patterns": ["career entirely in retail ops"],
            }
        )
    )
    calibration = brief._new_brief.facial_calibration
    assert calibration.expected_yes_rate_low == pytest.approx(0.08)
    assert calibration.band_source == "preflight"


# ---------------------------------------------------------------------------
# 4. Hypothesis confidence: Wilson lower bound, volume-aware
# ---------------------------------------------------------------------------


def test_wilson_lower_bound_table():
    from shared.search_memory import wilson_lower_bound

    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(0, 10) == 0.0
    # 2/6 and 2/400 no longer look identical.
    small = wilson_lower_bound(2, 6)
    large = wilson_lower_bound(2, 400)
    assert 0.0 < large < small < 1.0
    # Monotone in volume at fixed rate.
    assert wilson_lower_bound(20, 40) > wilson_lower_bound(2, 4)
    # Bounded.
    assert 0.0 <= wilson_lower_bound(40, 40) <= 1.0


def test_handoff_top_saves_preserve_unknown_confidence():
    """P6 lens fix (HIGH): a null judgment confidence must not launder into a
    fabricated 0.0 on the Chief-of-Staff surface — None survives and sorts
    after measured values."""
    from cloris.chief_of_staff.handoff import _extract_top_saves

    rows = [
        {
            "decision": "SAVE",
            "candidate_id": "unknown-conf",
            "confidence": None,
            "rationale": "parse failure upstream",
        },
        {
            "decision": "SAVE",
            "candidate_id": "low-conf",
            "confidence": 0.2,
            "rationale": "measured low",
        },
    ]

    saves = _extract_top_saves(rows)

    by_id = {s["candidate_id"]: s for s in saves}
    assert by_id["unknown-conf"]["confidence"] is None
    assert by_id["low-conf"]["confidence"] == pytest.approx(0.2)
    # Measured values sort ahead of unknowns — unknown is not "zero".
    assert [s["candidate_id"] for s in saves] == ["low-conf", "unknown-conf"]


def test_research_context_examples_preserve_unknown_confidence():
    """P6 lens fix (MED): market-intel research input carries null, not a
    fabricated 0.0, for a parse-failed confidence."""
    from market_intelligence.research_context import _build_candidate_examples

    examples = _build_candidate_examples(
        [
            {
                "decision": "SAVE",
                "candidate_name": "Jane Doe",
                "profile_url": "https://example.com/jane",
                "confidence": None,
                "rationale": "r",
            }
        ],
        [],
        [],
        None,
    )

    assert examples["saved_examples"][0]["confidence"] is None


def test_hypothesis_confidence_is_volume_aware_through_update():
    from shared.search_memory import update_search_memory

    class _StringStub:
        def __init__(self, saves: int, candidates: int) -> None:
            self.family_key = "fam"
            self.novelty_bucket = "edge_case"
            self.domain_lane = "general"
            self.boolean = '"x"'
            self.saves = ["cand"] * saves
            self.result_count = 100
            self.duplicates_count = 0
            self.candidates_count = candidates
            self.status = "done"
            self.retrieval_recipe = {}
            self.retrieval_hypothesis_ids = ["h1"]

    memory = update_search_memory(
        {},
        "test-project",
        [_StringStub(saves=2, candidates=6), _StringStub(saves=0, candidates=0)],
    )
    entry = memory["hypotheses"]["h1"]
    assert entry["status"] == "validated"
    from shared.search_memory import wilson_lower_bound

    assert entry["confidence"] == pytest.approx(
        wilson_lower_bound(entry["saves"], entry["candidate_count"])
    )
