"""CLO-208: claude-opus-5 vs the legacy full-eval wire contract.

Two live sessions (2026-08-18, 2026-08-19) lost 60 verdicts to PARSE_FAILURE
because Opus 5 echoes the prompt's NUMBERED capability-area list back as
annotated prose ("#3 Evals/benchmarks (adjacent, via ...)"), which the strict
V2 membership check rejects — and the parser swallowed the reason, so every
failure printed an opaque "PARSE_FAILURE". These tests lock the three fixes:
bounded area canonicalization, reason surfacing, and raw-text capture.
"""

import json

from shared.judgment.templates import (
    _canonicalize_capability_area,
    parse_full_evaluation_response,
)

AREAS = [
    "LLM post-training for coding capability",
    "Coding agents and scaffolds",
    "Evals, benchmarks, and RL environments for code",
    "Hands-on engineering and data infrastructure",
    "Technical leadership and delivery ownership",
]

# The exact area line claude-opus-5 wrote in the 2026-08-19 replay (two
# abbreviated areas) — must stay a PARSE_FAILURE: no guessing between areas.
OPUS_TWO_AREA_PROSE = (
    "#3 Evals/benchmarks (adjacent, via data-contributor role) "
    "and #4 hands-on engineering/data infrastructure"
)


def _response(area_line: str) -> str:
    return f"""STEP_1_MATCH: ADJACENT
STEP_1_AREA: {area_line}
STEP_1_EVIDENCE: Cited specific profile bullets.
STEP_2_DEPTH: USER
STEP_2_EVIDENCE: Consumption-shaped LLM work only.
STEP_3_TRANSFERABILITY: NOT_TRANSFERABLE
STEP_3_EVIDENCE: No qualifying transfer path.
STEP_1_RECENCY: CURRENT
STEP_4_LEVEL: ALIGNED
STEP_4_EVIDENCE: Senior scope demonstrated.
STEP_5_COHERENCE: UNCLEAR
STEP_5_DRIVER: direction mismatch
STEP_6_CALIBER: SOLID
STEP_6_EVIDENCE: Owned systems end to end.
CASE_FOR: Strong veteran engineer.
CASE_AGAINST: No personally built training or eval infrastructure.
DECISION: REJECT
CONFIDENCE: 0.78
REJECT_REASON: DEPTH_CONSUMER
OUTREACH_TIER: NONE
POST_SAVE_MODIFIER: NONE
REVIEW_REASON: NONE
SUMMARY: Excellent adjacent engineer below the builder bar.
"""


def _parse(raw: str):
    return parse_full_evaluation_response(
        raw,
        require_semantic_v2=True,
        capability_areas=AREAS,
        post_save_modifiers=[],
    )


def test_verbatim_area_parses():
    result = _parse(_response(AREAS[2]))
    assert result.decision == "REJECT"
    assert result.capability_area == AREAS[2]


def test_numbered_annotated_area_canonicalizes():
    line = "#3 Evals, benchmarks, and RL environments for code (via data-contributor role)"
    result = _parse(_response(line))
    assert result.decision == "REJECT"
    assert result.capability_area == AREAS[2]


def test_opus_two_area_prose_still_fails_with_named_reason():
    result = _parse(_response(OPUS_TWO_AREA_PROSE))
    assert result.decision == "PARSE_FAILURE"
    assert "invalid_capability_area" in result.summary
    assert result.raw_response  # raw text preserved for the capture path


def test_two_full_canonical_names_stay_ambiguous():
    line = f"{AREAS[2]} and {AREAS[3]}"
    result = _parse(_response(line))
    assert result.decision == "PARSE_FAILURE"


def test_canonicalizer_bounded_rules():
    fold = _canonicalize_capability_area
    assert fold("evals, benchmarks, and rl environments for code", AREAS) == AREAS[2]
    assert fold("2. Coding agents and scaffolds", AREAS) == AREAS[1]
    assert fold("(2) Coding agents and scaffolds", AREAS) == AREAS[1]
    assert fold("Area 2: Coding agents and scaffolds", AREAS) == AREAS[1]
    assert fold("Coding  agents   and scaffolds", AREAS) == AREAS[1]
    # Abbreviations never coerce.
    assert fold("Evals/benchmarks", AREAS) == "Evals/benchmarks"
    # Containment of two canonical names never coerces.
    two = f"{AREAS[0]} plus {AREAS[1]}"
    assert fold(two, AREAS) == two
    # Empty/None pass through.
    assert fold(None, AREAS) is None
    assert fold("", AREAS) == ""


def test_full_parse_failure_raw_capture(tmp_path, monkeypatch):
    import shared.judger as judger_mod

    monkeypatch.setattr(
        judger_mod,
        "current_llm_usage_log_path",
        lambda: tmp_path / "llm-attempts.jsonl",
    )
    judger_mod._record_full_parse_failure(
        candidate_name="Murilo Perrone",
        reason="PARSE_FAILURE — invalid_capability_area: #3 ...",
        raw="STEP_1_MATCH: ADJACENT\n...",
    )
    target = tmp_path / "full_parse_failures.jsonl"
    record = json.loads(target.read_text().splitlines()[0])
    assert record["candidate_name"] == "Murilo Perrone"
    assert "invalid_capability_area" in record["reason"]
    assert record["raw"].startswith("STEP_1_MATCH:")
