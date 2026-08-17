"""Tests for judger prompt generation and brief loading.

Run with: python -m pytest test_extractors.py -v
"""

import json
import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

import shared.extractors as extractors
from shared.schemas import CandidateSnippet, CandidateProfileSummary
from shared.extractors import extract_snippet_from_card_innertext
from shared.judger import _build_facial_system, _build_full_system, facial_judge, full_judge
from shared.brief_loader import load_brief, Brief
from linkedin.judgment_templates import assemble_facial_system, assemble_full_evaluation_system


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
BRAZIL_BRIEF_PATH = str(ROOT / "config" / "FDL-Brazil" / "brief-brazil-real.json")
HEAD_AI_BRIEF_PATH = str(ROOT / "config" / "Head-of-Applied-AI-Lab" / "brief-head-ai-lab-real.json")
HEAD_AI_V2_BRIEF_PATH = str(ROOT / "config" / "brief-head-ai-lab-nyc-v2.json")

for _path in (BRAZIL_BRIEF_PATH, HEAD_AI_BRIEF_PATH, HEAD_AI_V2_BRIEF_PATH):
    if not Path(_path).is_file():
        pytest.skip(
            "Optional sourcing brief JSON not found under config/ — add local fixtures to run this module.",
            allow_module_level=True,
        )

BRAZIL_BRIEF = load_brief(BRAZIL_BRIEF_PATH)
HEAD_AI_BRIEF = load_brief(HEAD_AI_BRIEF_PATH)
HEAD_AI_V2_BRIEF = load_brief(HEAD_AI_V2_BRIEF_PATH)


def _make_snippet(**kwargs) -> CandidateSnippet:
    defaults = {
        "name": "Test Person",
        "headline": "",
        "current_title": "",
        "current_company": "",
        "location": "Sao Paulo, Brazil",
        "education_snippet": "",
        "profile_url": "/talent/profile/test",
        "source_string_id": 1,
        "source_string_name": "test",
        "page": 1,
        "result_rank": 1,
    }
    defaults.update(kwargs)
    return CandidateSnippet(**defaults)


# ---------------------------------------------------------------------------
# Task 1: Confirm hard_filters is gone
# ---------------------------------------------------------------------------

def test_orchestrator_does_not_import_hard_filters():
    """Verify orchestrator.py has no reference to hard_filters."""
    source = (Path(__file__).parent.parent / "linkedin" / "orchestrator.py").read_text()
    assert "hard_filters" not in source
    assert "hard_filter" not in source


def test_hard_filters_module_deleted():
    """Verify hard_filters.py no longer exists."""
    assert not (Path(__file__).parent.parent / "hard_filters.py").exists()


# ---------------------------------------------------------------------------
# Judger prompt generation from brief
# ---------------------------------------------------------------------------

def test_github_portfolio_yes_defaults_are_domain_neutral():
    """Brief-schema GitHub facial defaults must not hardcode a frontier-AI vertical."""
    from shared.brief_schema import (
        Brief,
        CapabilityArea,
        DepthDistinction,
        FacialCalibration,
        MarketDensity,
    )

    brief = Brief(
        role_title="Platform Engineer",
        role_level="IC5",
        role_summary="Owns core infrastructure.",
        geography="Remote",
        linkedin_project="test",
        capability_areas=[
            CapabilityArea(
                name="Infra",
                description="Core platform work",
                builder_signals=["ships production systems"],
                user_signals=["tutorial repos only"],
            )
        ],
        depth_distinction=DepthDistinction(
            builder_definition="Ships production infrastructure systems.",
            user_definition="Uses infrastructure built by others.",
            edge_case_guidance="Default to full evaluation when ambiguous.",
        ),
        non_fit_patterns=[],
        employer_signal_rules=[],
        minimum_years_experience=5,
        minimum_bar_description="Has shipped infra at scale.",
        facial_calibration=FacialCalibration(
            expected_yes_rate_low=0.25,
            expected_yes_rate_high=0.55,
            fast_exit_patterns=["Entirely unrelated work"],
            trajectory_yes_patterns=["Infra ownership progression"],
            trajectory_ambiguous_patterns=["Mixed signal"],
            trajectory_no_patterns=["Unrelated career"],
        ),
        market_density=MarketDensity.MODERATE,
    )
    block = brief.github_portfolio_yes_block()

    assert "widely-depended-on open source projects in the role's domain" in block
    assert "huggingface" not in block.lower()
    assert "frontier" not in block.lower()


def test_facial_prompt_includes_brazil_brief_content():
    prompt = _build_facial_system(BRAZIL_BRIEF)
    assert "fdl-brazil" in prompt.lower() or "frontier data lead" in prompt.lower()
    assert "Post-Training Data Engineer" in prompt
    assert "Coding Agent Evaluator" in prompt
    assert "Annotation Worker" in prompt
    assert "minimum_bar" in prompt.lower() or "BUILDING model training" in prompt


def test_full_prompt_includes_brazil_brief_content():
    prompt = _build_full_system(BRAZIL_BRIEF)
    assert "Post-Training Data Engineer" in prompt
    assert "save_signals" in prompt.lower() or "Save signals" in prompt
    assert "caution signals" in prompt.lower() or "Caution signals" in prompt
    assert "Fintech ML Engineer" in prompt
    assert "experience_floor" in prompt.lower() or "Experience Floor" in prompt


def test_facial_prompt_includes_head_ai_brief():
    """Proves the system is brief-agnostic — different brief, different prompt."""
    prompt = _build_facial_system(HEAD_AI_BRIEF)
    assert "head" in prompt.lower() or "applied ai" in prompt.lower()
    # This brief has different structure — should still build without error


def test_full_prompt_includes_head_ai_brief():
    prompt = _build_full_system(HEAD_AI_BRIEF)
    # Head AI brief has clear_skips_from_review as dicts with "pattern" and "reason"
    assert "Solutions Architect" in prompt or "solutions architect" in prompt.lower()


def test_prompts_not_hardcoded():
    """Ensure prompts don't contain the old hardcoded role description."""
    facial = _build_facial_system(BRAZIL_BRIEF)
    full = _build_full_system(BRAZIL_BRIEF)
    # These phrases were in the old hardcoded prompts
    assert "partner with researchers at OpenAI, Anthropic, and DeepMind" not in full
    assert "MSc/PhD from strong research institution" not in full


# ---------------------------------------------------------------------------
# Brief loader tests (Task 3)
# ---------------------------------------------------------------------------

def test_brazil_brief_loads():
    brief = load_brief(BRAZIL_BRIEF_PATH)
    assert isinstance(brief, Brief)
    assert brief.id == "fdl-brazil"
    assert "Frontier Data Lead" in brief.role_description
    assert brief.kit_url.startswith("https://search-kit-library.vercel.app/kit/")
    assert len(brief.archetypes) >= 5
    assert brief.archetypes[0]["name"] == "Post-Training Data Engineer"
    assert brief.permanent_filters.get("Location") == "Brazil"


def test_head_ai_brief_loads():
    brief = load_brief(HEAD_AI_BRIEF_PATH)
    assert isinstance(brief, Brief)
    assert brief.id == "head-applied-ai-lab-nyc"
    assert "mini-cto" in brief.role_description.lower() or "applied ai" in brief.role_description.lower()
    assert brief.kit_url.startswith("https://search-kit-library.vercel.app/kit/")
    assert brief.linkedin_project == "Head of Applied AI Lab"
    assert brief.linkedin_project_id == "3000000006"


def test_head_ai_v2_brief_loads():
    brief = load_brief(HEAD_AI_V2_BRIEF_PATH)
    assert isinstance(brief, Brief)
    assert brief.has_v2_schema is True
    assert brief.role_title == "Head of Applied AI Lab"
    assert brief.linkedin_project == "Head of Applied AI Lab"
    assert brief.linkedin_project_id == "3000000006"
    assert brief.additional_search_terms
    assert (
        "capital-markets or broader financial-institution workflow exposure with applied genai depth"
        in brief.additional_search_terms
    )
    assert (
        "document or knowledge workflows tied to financial-institution artifacts rather than generic legal-tech"
        in brief.additional_search_terms
    )
    assert any("Capital markets" in priority or "capital-markets" in priority for priority in brief.search_priorities)
    assert any("Payments" in priority or "payments" in priority for priority in brief.search_priorities)
    assert any("first 8 strings" in instruction.lower() for instruction in brief.instructions)
    assert any("first 10-12 strings" in instruction.lower() for instruction in brief.instructions)
    assert brief.market_density == "sparse"


def test_head_ai_v2_full_prompt_includes_market_intel_calibration():
    prompt = assemble_full_evaluation_system(HEAD_AI_V2_BRIEF._new_brief)
    assert "Ashish Garg" in prompt
    assert "Pratik Shah" in prompt
    assert "Executive Director" in prompt
    assert "Principal Architect" in prompt


def test_head_ai_v2_facial_prompt_includes_updated_trajectory_patterns():
    prompt = assemble_facial_system(HEAD_AI_V2_BRIEF._new_brief)
    assert "Career progression from big-bank, market-infrastructure, market-data, or top-tech builder roles" in prompt
    assert "Startup CTO or co-founder at a small fintech, regtech, payments, market-data, insurance, or other BFSI-serving company with obvious hands-on language" in prompt
    assert "Trajectory centers on surveillance, compliance-tech, or risk operations and shows no credible recent AI or GenAI ownership" in prompt
    assert "field CTO, customer engineering, solutions, or vendor advisory leadership without direct production build ownership" in prompt


def test_both_briefs_have_kit_url():
    brazil = load_brief(BRAZIL_BRIEF_PATH)
    head_ai = load_brief(HEAD_AI_BRIEF_PATH)
    assert brazil.kit_url.startswith("https://")
    assert head_ai.kit_url.startswith("https://")
    assert "search-kit-library" in brazil.kit_url
    assert "search-kit-library" in head_ai.kit_url


def test_both_briefs_archetypes_normalized():
    brazil = load_brief(BRAZIL_BRIEF_PATH)
    head_ai = load_brief(HEAD_AI_BRIEF_PATH)

    for brief in [brazil, head_ai]:
        assert isinstance(brief.archetypes, list)
        assert len(brief.archetypes) > 0
        for arch in brief.archetypes:
            assert "name" in arch
            assert "pattern" in arch


def test_clear_skips_flattened():
    """Head of AI Lab brief has dicts; Brazil has strings. Both should flatten."""
    brazil = load_brief(BRAZIL_BRIEF_PATH)
    head_ai = load_brief(HEAD_AI_BRIEF_PATH)

    for brief in [brazil, head_ai]:
        assert isinstance(brief.clear_skips_from_review, list)
        for item in brief.clear_skips_from_review:
            assert isinstance(item, str)


def test_minimum_bar_is_string():
    """minimum_bar should always be a string (Head AI Lab has it as dict)."""
    brazil = load_brief(BRAZIL_BRIEF_PATH)
    head_ai = load_brief(HEAD_AI_BRIEF_PATH)
    assert isinstance(brazil.minimum_bar, str)
    assert isinstance(head_ai.minimum_bar, str)
    assert len(brazil.minimum_bar) > 50
    assert len(head_ai.minimum_bar) > 50


def test_raw_dict_preserved():
    brazil = load_brief(BRAZIL_BRIEF_PATH)
    assert brazil.raw.get("name") == "fdl-brazil"
    head_ai = load_brief(HEAD_AI_BRIEF_PATH)
    assert head_ai.raw.get("brief_id") == "head-applied-ai-lab-nyc"


def test_pipeline_import_works():
    """Verify orchestrator module imports cleanly."""
    from linkedin.orchestrator import Pipeline
    assert Pipeline is not None


def test_single_card_extractor_builds_snippet():
    with patch("shared.extractors.cheap_llm", return_value={
        "name": "Ada Lovelace",
        "headline": "ML Engineer",
        "current_title": "ML Engineer",
        "current_company": "Analytical Engines",
        "location": "London",
        "education_snippet": "University of London",
        "profile_url": "/talent/profile/ada",
        "experience_entries": ["ML Engineer at Analytical Engines (2024-Present)"],
    }):
        snippet = extract_snippet_from_card_innertext(
            "Select Ada Lovelace\nAda Lovelace\nML Engineer",
            string_id=7,
            string_name="test string",
            page=2,
            result_rank=4,
        )

    assert snippet is not None
    assert snippet.name == "Ada Lovelace"
    assert snippet.profile_url == "/talent/profile/ada"
    assert snippet.source_string_id == 7
    assert snippet.source_string_name == "test string"
    assert snippet.page == 2
    assert snippet.result_rank == 4


def test_single_card_extractor_uses_dom_hints_when_model_misses_name():
    with patch("shared.extractors.cheap_llm", return_value={
        "name": "",
        "headline": "Principal AI Engineer at Example Bank",
        "current_title": "",
        "current_company": "",
        "location": "New York, New York, United States",
        "education_snippet": "",
        "profile_url": "",
        "experience_entries": ["Principal AI Engineer at Example Bank (2024-Present)"],
    }):
        snippet = extract_snippet_from_card_innertext(
            "Select Ada Lovelace\nAda Lovelace\nPrincipal AI Engineer at Example Bank",
            string_id=1,
            string_name="test",
            page=1,
            result_rank=1,
            dom_name="Ada Lovelace",
            dom_url="/talent/profile/ada",
        )

    assert snippet is not None
    assert snippet.name == "Ada Lovelace"
    assert snippet.profile_url == "/talent/profile/ada"
    assert snippet.current_title == "Principal AI Engineer"
    assert snippet.current_company == "Example Bank"


def test_single_card_extractor_falls_back_to_heuristics_for_invalid_payload():
    with patch("shared.extractors.cheap_llm", return_value="not-json"):
        snippet = extract_snippet_from_card_innertext(
            "\n".join([
                "Select Test Person",
                "Test Person",
                "Principal AI Engineer",
                "New York City Metropolitan Area · Financial Services",
                "Experience",
                "Profile experience",
                "Principal AI Engineer at Example Bank · 2022 - Present",
                "Education",
                "Profile education",
                "Example University, MS Computer Science",
            ]),
            string_id=1,
            string_name="test",
            page=1,
            result_rank=1,
        )

    assert snippet is not None
    assert snippet.name == "Test Person"
    assert snippet.current_title == "Principal AI Engineer"
    assert snippet.current_company == "Example Bank"
    assert snippet.location == "New York City Metropolitan Area"
    assert snippet.education_snippet == "Example University, MS Computer Science"


def test_single_card_extractor_returns_none_for_unrecoverable_payload():
    with patch("shared.extractors.cheap_llm", return_value="not-json"):
        snippet = extract_snippet_from_card_innertext(
            "Experience\nSave to pipeline",
            string_id=1,
            string_name="test",
            page=1,
            result_rank=1,
        )

    assert snippet is None


# ---------------------------------------------------------------------------
# Judgment parse-failure hardening (old-brief paths)
# ---------------------------------------------------------------------------

def _make_summary(**kwargs) -> CandidateProfileSummary:
    defaults = {
        "name": "Test Person",
        "profile_url": "/talent/profile/test",
        "headline": "Engineer",
    }
    defaults.update(kwargs)
    return CandidateProfileSummary(**defaults)


def test_facial_missing_decision_parse_failure():
    """Old-brief facial: missing decision key -> PARSE_FAILURE."""
    with patch("shared.judger.opus_llm_cached", return_value={}):
        result = facial_judge(_make_snippet(), BRAZIL_BRIEF)
    assert result.decision == "PARSE_FAILURE"
    assert result.confidence == 0.0
    assert "PARSE_FAILURE" in result.rationale


def test_facial_garbage_decision_parse_failure():
    """Old-brief facial: unrecognized decision value -> PARSE_FAILURE."""
    with patch("shared.judger.opus_llm_cached", return_value={"decision": "YOLO"}):
        result = facial_judge(_make_snippet(), BRAZIL_BRIEF)
    assert result.decision == "PARSE_FAILURE"
    assert result.confidence == 0.0


def test_facial_nondict_result_parse_failure():
    """Old-brief facial: non-dict LLM result -> PARSE_FAILURE."""
    with patch("shared.judger.opus_llm_cached", return_value="just a string"):
        result = facial_judge(_make_snippet(), BRAZIL_BRIEF)
    assert result.decision == "PARSE_FAILURE"
    assert result.confidence == 0.0


def test_full_missing_decision_parse_failure():
    """Old-brief full: missing decision key -> PARSE_FAILURE."""
    with patch("shared.judger.opus_llm_cached", return_value={}):
        result = full_judge(_make_summary(), BRAZIL_BRIEF)
    assert result.decision == "PARSE_FAILURE"
    assert result.confidence == 0.0
    assert "PARSE_FAILURE" in result.rationale


def test_full_garbage_decision_parse_failure():
    """Old-brief full: unrecognized decision value -> PARSE_FAILURE."""
    with patch("shared.judger.opus_llm_cached", return_value={"decision": "MAYBE"}):
        result = full_judge(_make_summary(), BRAZIL_BRIEF)
    assert result.decision == "PARSE_FAILURE"
    assert result.confidence == 0.0


def test_full_nondict_result_parse_failure():
    """Old-brief full: non-dict LLM result -> PARSE_FAILURE."""
    with patch("shared.judger.opus_llm_cached", return_value=42):
        result = full_judge(_make_summary(), BRAZIL_BRIEF)
    assert result.decision == "PARSE_FAILURE"
    assert result.confidence == 0.0


def test_facial_malformed_confidence_safe():
    """Old-brief facial: non-numeric confidence -> default, not exception."""
    with patch("shared.judger.opus_llm_cached", return_value={
        "decision": "FACIAL_YES", "confidence": "high", "rationale": "ok"
    }):
        result = facial_judge(_make_snippet(), BRAZIL_BRIEF)
    assert result.decision == "FACIAL_YES"
    assert result.confidence == 0.5  # default fallback


def test_full_malformed_confidence_safe():
    """Old-brief full: non-numeric confidence -> default, not exception."""
    with patch("shared.judger.opus_llm_cached", return_value={
        "decision": "SAVE", "confidence": "???", "rationale": "match"
    }):
        result = full_judge(_make_summary(), BRAZIL_BRIEF)
    assert result.decision == "SAVE"
    assert result.confidence == 0.5  # default fallback


def test_facial_opus_exception_judgment_failure():
    """Old-brief facial: opus_llm_cached exception -> JUDGMENT_FAILURE."""
    with patch("shared.judger.opus_llm_cached", side_effect=RuntimeError("API timeout")):
        result = facial_judge(_make_snippet(), BRAZIL_BRIEF)
    assert result.decision == "JUDGMENT_FAILURE"
    assert result.confidence == 0.0


def test_full_opus_exception_judgment_failure():
    """Old-brief full: opus_llm_cached exception -> JUDGMENT_FAILURE."""
    with patch("shared.judger.opus_llm_cached", side_effect=RuntimeError("API timeout")):
        result = full_judge(_make_summary(), BRAZIL_BRIEF)
    assert result.decision == "JUDGMENT_FAILURE"
    assert result.confidence == 0.0


def test_facial_none_result_parse_failure():
    """Old-brief facial: None result -> PARSE_FAILURE."""
    with patch("shared.judger.opus_llm_cached", return_value=None):
        result = facial_judge(_make_snippet(), BRAZIL_BRIEF)
    assert result.decision == "PARSE_FAILURE"
    assert result.confidence == 0.0


def test_is_failure_decision_helper():
    """is_failure_decision returns True for failures, False for real decisions."""
    from shared.judger import is_failure_decision
    assert is_failure_decision("PARSE_FAILURE") is True
    assert is_failure_decision("JUDGMENT_FAILURE") is True
    assert is_failure_decision("FACIAL_YES") is False
    assert is_failure_decision("FACIAL_NO") is False
    assert is_failure_decision("SAVE") is False
    assert is_failure_decision("REJECT") is False
    assert is_failure_decision("FACIAL_SKIP") is False


# ---------------------------------------------------------------------------
# Parser-level regression tests
# ---------------------------------------------------------------------------

def test_parse_facial_malformed_returns_parse_failure():
    """parse_facial_response with garbage input -> PARSE_FAILURE decision."""
    from linkedin.judgment_templates import parse_facial_response
    result = parse_facial_response("totally garbage response with no decision")
    assert result.decision == "PARSE_FAILURE"


def test_parse_full_malformed_returns_parse_failure():
    """parse_full_evaluation_response with garbage input -> PARSE_FAILURE decision."""
    from linkedin.judgment_templates import parse_full_evaluation_response
    result = parse_full_evaluation_response("totally garbage response")
    assert result.decision == "PARSE_FAILURE"


# ---------------------------------------------------------------------------
# V2 path regression tests
# ---------------------------------------------------------------------------

def test_v2_facial_parse_failure_not_skip():
    """V2 facial: parser failure -> PARSE_FAILURE, not FACIAL_SKIP."""
    with patch("shared.judger.opus_llm_cached", return_value="garbage with no decision line"):
        result = facial_judge(_make_snippet(), HEAD_AI_BRIEF)
    assert result.decision == "PARSE_FAILURE"
    assert result.decision != "FACIAL_SKIP"
    assert result.confidence == 0.0


def test_v2_full_parse_failure_not_reject():
    """V2 full: parser failure -> PARSE_FAILURE, not REJECT."""
    with patch("shared.judger.opus_llm_cached", return_value="garbage with no structure"):
        result = full_judge(_make_summary(), HEAD_AI_BRIEF)
    assert result.decision == "PARSE_FAILURE"
    assert result.decision != "REJECT"


def test_github_full_parse_failure_not_reject():
    """GitHub full: parser failure -> PARSE_FAILURE, not REJECT.

    Tests via the parser directly since github_full_judge requires V2 brief.
    The parser is re-exported from linkedin.judgment_templates.
    """
    from github.judgment_templates import parse_full_evaluation_response
    result = parse_full_evaluation_response("garbage with no structure at all")
    assert result.decision == "PARSE_FAILURE"
    assert result.decision != "REJECT"


def test_facial_valid_yes_passes():
    """Old-brief facial: valid FACIAL_YES passes through normally."""
    with patch("shared.judger.opus_llm_cached", return_value={
        "decision": "FACIAL_YES", "path": "eng", "confidence": 0.8, "rationale": "looks good"
    }):
        result = facial_judge(_make_snippet(), BRAZIL_BRIEF)
    assert result.decision == "FACIAL_YES"
    assert result.confidence == 0.8
    assert result.rationale == "looks good"


def test_full_valid_save_passes():
    """Old-brief full: valid SAVE passes through normally."""
    with patch("shared.judger.opus_llm_cached", return_value={
        "decision": "SAVE", "path": "eng", "confidence": 0.7, "rationale": "strong match"
    }):
        result = full_judge(_make_summary(), BRAZIL_BRIEF)
    assert result.decision == "SAVE"
    assert result.confidence == 0.7
    assert result.rationale == "strong match"
