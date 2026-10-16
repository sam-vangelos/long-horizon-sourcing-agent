"""Researcher Slice 5 — evaluator coverage.

Pins:
- Facial fast-exit gates fire before any LLM call when h_index or
  papers_in_window is below the resolved floor; rationale is recruiter-
  readable (no engineer vocab).
- Surviving facial snippets receive a single batched LLM call.
- Full evaluation produces an OpusDecision shape that round-trips
  through `extract_save_reason_and_confidence` (the wire contract for
  every module per Spec Opinion 6).
- llm_caller injection lets tests run without real Opus.
- Templates render brief context (capability areas, depth_distinction)
  without leaking engineer vocab.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from researcher.judgment_templates import (
    assemble_facial_system,
    assemble_full_evaluation_system,
    fast_exit_rationale_for_h_index,
    fast_exit_rationale_for_papers,
    render_facial_user_prompt,
    render_full_user_prompt,
)
from researcher.schemas import (
    ResearcherCandidate,
    ResearcherPaper,
    ResearcherSnippet,
)
from shared.judger import researcher_facial_judge_batch, researcher_full_judge
from shared.runtime_state.read_models import extract_save_reason_and_confidence


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _stub_brief() -> SimpleNamespace:
    capability_areas = [
        SimpleNamespace(
            name="Post-training research",
            description="Original work on RLHF / DPO / SFT.",
        ),
        SimpleNamespace(
            name="Inference systems",
            description="Quantization / distillation / serving.",
        ),
    ]
    depth_distinction = SimpleNamespace(
        builder_definition="First-author publications at canonical venues.",
        user_definition="Cites without publishing.",
        edge_case_guidance="Borderline = full eval.",
    )
    return SimpleNamespace(
        role_title="Frontier-lab Researcher",
        capability_areas=capability_areas,
        depth_distinction=depth_distinction,
    )


def _make_snippet(
    *,
    name: str = "Jane R.",
    h_index: int = 12,
    papers_in_window: int = 5,
    affiliation: str = "MIT (US)",
    top_papers: list[str] | None = None,
    profile_url: str = "https://orcid.org/0000-0001-2345-6789",
) -> ResearcherSnippet:
    return ResearcherSnippet(
        name=name,
        current_affiliation=affiliation,
        h_index=h_index,
        citation_count=h_index * 30,
        papers_in_window=papers_in_window,
        top_paper_titles=top_papers or ["Survey of RLHF (NeurIPS 2024)"],
        arxiv_categories=["cs.LG", "cs.CL"],
        profile_url=profile_url,
    )


def _make_candidate(
    *,
    name: str = "Jane R.",
    h_index: int = 14,
    papers_in_window: int = 6,
) -> ResearcherCandidate:
    return ResearcherCandidate(
        author_id="A1234",
        orcid="0000-0001-2345-6789",
        name=name,
        affiliations=["MIT (US)"],
        top_papers=[
            ResearcherPaper(
                title="Survey of RLHF",
                venue="NeurIPS",
                year=2024,
                citation_count=120,
                is_first_author=True,
            )
        ],
        h_index=h_index,
        citation_count=300,
        works_count=22,
        papers_in_window=papers_in_window,
        profile_url="https://orcid.org/0000-0001-2345-6789",
    )


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def test_facial_system_prompt_includes_role_and_capability_areas() -> None:
    brief = _stub_brief()
    system = assemble_facial_system(brief)
    assert "Frontier-lab Researcher" in system
    assert "Post-training research" in system
    assert "Inference systems" in system


def test_full_system_prompt_includes_depth_distinction() -> None:
    brief = _stub_brief()
    system = assemble_full_evaluation_system(brief)
    assert "First-author publications" in system
    assert "Builder:" in system
    assert "User:" in system


def test_facial_user_prompt_renders_snippet_fields() -> None:
    snippet = _make_snippet(
        name="Wei Wang",
        affiliation="Stanford (US)",
        h_index=9,
        papers_in_window=4,
        top_papers=["Inference systems for LLMs", "Quantized fine-tuning"],
    )
    body = render_facial_user_prompt(snippet)
    assert "Wei Wang" in body
    assert "Stanford (US)" in body
    assert "h-index: 9" in body
    assert "Papers in window: 4" in body
    assert "Inference systems for LLMs" in body
    assert "cs.LG, cs.CL" in body


def test_full_user_prompt_renders_top_papers_with_first_author_marker() -> None:
    candidate = _make_candidate()
    body = render_full_user_prompt(candidate)
    assert "Jane R." in body
    assert "h-index: 14" in body
    assert "Survey of RLHF" in body
    assert "[first author]" in body
    assert "NeurIPS" in body
    assert "(2024)" in body
    assert "cited 120x" in body


def test_facial_prompt_avoids_engineer_vocab_leak() -> None:
    """Spec Opinion 7: rationale + prompts should not leak
    `papers_in_window`, `h_index_floor`, etc. as variable names.
    """

    snippet = _make_snippet()
    body = render_facial_user_prompt(snippet)
    # The prompt is allowed to label fields ("h-index:", "Papers in window:")
    # but the engineer-suffix `_floor` / `_count` should never leak.
    assert "_floor" not in body
    assert "h_index:" not in body
    assert "papers_in_window=" not in body


# ---------------------------------------------------------------------------
# Fast-exit rationale builders
# ---------------------------------------------------------------------------


def test_fast_exit_papers_rationale_renders_recruiter_copy_for_nlp() -> None:
    rationale = fast_exit_rationale_for_papers(
        papers_in_window=1,
        papers_in_window_floor=3,
        papers_in_window_months=24,
        discipline="nlp",
    )
    assert (
        rationale
        == "Skipped — only 1 paper in the last 24 months, below the NLP minimum of 3."
    )


def test_fast_exit_papers_rationale_uses_field_defaults_label_when_no_discipline() -> None:
    rationale = fast_exit_rationale_for_papers(
        papers_in_window=0,
        papers_in_window_floor=1,
        papers_in_window_months=36,
        discipline="",
    )
    assert "field defaults" in rationale
    assert "0 papers in the last 36 months" in rationale


def test_fast_exit_h_index_rationale_renders_recruiter_copy() -> None:
    rationale = fast_exit_rationale_for_h_index(
        h_index=2,
        h_index_floor=8,
        discipline="ml_general",
    )
    assert rationale == "Skipped — h-index 2, below the general ML minimum of 8."


def test_fast_exit_rationale_avoids_engineer_vocab() -> None:
    """No `_floor`, no equation form, no field-name leakage."""

    rationale = fast_exit_rationale_for_papers(
        papers_in_window=1,
        papers_in_window_floor=3,
        papers_in_window_months=24,
        discipline="nlp",
    )
    assert "_floor" not in rationale
    assert "papers_in_window" not in rationale
    assert "<" not in rationale
    assert "==" not in rationale


# ---------------------------------------------------------------------------
# Researcher facial judge batch
# ---------------------------------------------------------------------------


def test_researcher_facial_batch_fast_exits_below_papers_floor() -> None:
    """Snippet with papers_in_window=1 + universal floor=1 should NOT
    fast-exit; with floor=3 (NLP discipline) it should."""

    brief = _stub_brief()
    snippet = _make_snippet(papers_in_window=1, h_index=20)

    # No discipline + no override → universal floor papers_in_window=1
    # The snippet has papers_in_window=1 which equals (not below) the
    # floor; should NOT fast-exit. We expect the LLM to be called.
    llm_calls: list[tuple[str, str]] = []

    def llm_caller(system: str, user: str) -> dict:
        llm_calls.append((system, user))
        return {
            "decision": "FACIAL_YES",
            "rationale": "Recent work in capability area",
            "confidence": 0.8,
        }

    decisions = researcher_facial_judge_batch(
        [snippet],
        brief=brief,
        source_config={},
        llm_caller=llm_caller,
    )
    assert decisions[0].decision == "FACIAL_YES"
    assert len(llm_calls) == 1


def test_researcher_facial_batch_fast_exits_below_nlp_papers_floor() -> None:
    """With nlp discipline (floor=3), papers_in_window=1 fast-exits."""

    brief = _stub_brief()
    snippet = _make_snippet(papers_in_window=1, h_index=20, name="Jane")

    llm_calls: list[tuple[str, str]] = []

    def llm_caller(system: str, user: str) -> dict:
        llm_calls.append((system, user))
        return {"decision": "FACIAL_YES"}

    decisions = researcher_facial_judge_batch(
        [snippet],
        brief=brief,
        source_config={"discipline": "nlp"},
        llm_caller=llm_caller,
    )
    assert decisions[0].decision == "FACIAL_NO"
    assert decisions[0].path == "fast_exit:papers_in_window"
    assert "1 paper in the last 24 months" in decisions[0].rationale
    assert "NLP minimum of 3" in decisions[0].rationale
    assert len(llm_calls) == 0  # no LLM call for fast-exit


def test_researcher_facial_batch_fast_exits_below_h_index_floor() -> None:
    brief = _stub_brief()
    snippet = _make_snippet(h_index=2, papers_in_window=10)

    decisions = researcher_facial_judge_batch(
        [snippet],
        brief=brief,
        source_config={"discipline": "nlp"},
        llm_caller=lambda _s, _u: {"decision": "FACIAL_YES"},
    )
    assert decisions[0].decision == "FACIAL_NO"
    assert decisions[0].path == "fast_exit:h_index"
    assert "h-index 2" in decisions[0].rationale


def test_researcher_facial_batch_mixes_fast_exit_and_llm_results() -> None:
    """Two snippets — one fast-exits, one survives to the LLM. The
    output preserves input order.
    """

    brief = _stub_brief()
    snippet_fast = _make_snippet(name="Low", papers_in_window=0)
    snippet_llm = _make_snippet(name="High", papers_in_window=10, h_index=20)

    def llm_caller(system: str, user: str) -> dict:
        return {
            "decision": "FACIAL_BORDERLINE",
            "rationale": "Adjacent capability",
            "confidence": 0.55,
        }

    decisions = researcher_facial_judge_batch(
        [snippet_fast, snippet_llm],
        brief=brief,
        source_config={},
        llm_caller=llm_caller,
    )
    assert len(decisions) == 2
    assert decisions[0].decision == "FACIAL_NO"
    assert decisions[0].candidate_name == "Low"
    assert decisions[1].decision == "FACIAL_BORDERLINE"
    assert decisions[1].candidate_name == "High"


def test_researcher_facial_batch_handles_llm_failure_per_snippet() -> None:
    brief = _stub_brief()
    snippet = _make_snippet(papers_in_window=10, h_index=20)

    def failing_llm(_s: str, _u: str) -> dict:
        raise RuntimeError("opus down")

    decisions = researcher_facial_judge_batch(
        [snippet],
        brief=brief,
        source_config={},
        llm_caller=failing_llm,
    )
    # Failure decisions don't decision="FACIAL_YES" — they get a failure marker
    # via judgment_failure_decision; the orchestrator routes them as transient.
    assert len(decisions) == 1


# ---------------------------------------------------------------------------
# Researcher full judge — Spec Opinion 6 wire contract
# ---------------------------------------------------------------------------


def test_researcher_full_judge_produces_opus_decision_shape() -> None:
    brief = _stub_brief()
    candidate = _make_candidate()

    def llm_caller(system: str, user: str) -> dict:
        return {
            "decision": "SAVE",
            "path": "first_author_at_canonical_venue",
            "confidence": 0.91,
            "rationale": (
                "First-author at NeurIPS 2024 on RLHF reward modeling; "
                "h-index 14 aligns with post-training capability area."
            ),
        }

    decision = researcher_full_judge(candidate, brief=brief, llm_caller=llm_caller)
    assert decision.stage == "full"
    assert decision.decision == "SAVE"
    assert decision.path == "first_author_at_canonical_venue"
    assert decision.confidence == 0.91
    assert "First-author at NeurIPS 2024" in decision.rationale
    assert decision.candidate_name == "Jane R."
    assert decision.profile_url == "https://orcid.org/0000-0001-2345-6789"


def test_full_decision_round_trips_through_extract_save_reason_and_confidence() -> None:
    """The wire contract per Spec Opinion 6: rationale + confidence
    written under terminal_payload["full_decision"] are read back by
    `extract_save_reason_and_confidence` source-agnostically.
    """

    brief = _stub_brief()
    candidate = _make_candidate()

    decision = researcher_full_judge(
        candidate,
        brief=brief,
        llm_caller=lambda _s, _u: {
            "decision": "SAVE",
            "rationale": "Strong original research at canonical venues.",
            "confidence": 0.88,
        },
    )

    # Simulate the orchestrator's terminal_payload write path.
    terminal_payload = {"full_decision": decision.to_dict()}
    save_reason, confidence = extract_save_reason_and_confidence(terminal_payload)

    assert save_reason == "Strong original research at canonical venues."
    assert confidence == 0.88


def test_researcher_full_judge_handles_llm_string_response() -> None:
    """When the llm_caller returns a raw JSON string (the default LLM
    client does this with expect_json=False), the parser handles it.
    """

    brief = _stub_brief()
    candidate = _make_candidate()

    def llm_caller(system: str, user: str) -> str:
        return json.dumps(
            {
                "decision": "INFERENTIAL_SAVE",
                "rationale": "Adjacent methods.",
                "confidence": 0.6,
            }
        )

    decision = researcher_full_judge(candidate, brief=brief, llm_caller=llm_caller)
    assert decision.decision == "INFERENTIAL_SAVE"


def test_researcher_full_judge_handles_malformed_llm_output() -> None:
    brief = _stub_brief()
    candidate = _make_candidate()

    decision = researcher_full_judge(
        candidate,
        brief=brief,
        llm_caller=lambda _s, _u: "not json at all",
    )
    # Failure decisions exist as a known shape; we just verify it
    # doesn't raise. The orchestrator handles routing.
    assert decision.stage == "full"


def test_researcher_judges_require_brief() -> None:
    """No global brief, no kwarg → RuntimeError per Slice 5 contract."""

    import pytest

    import shared.judger as judger

    judger._brief = None

    with pytest.raises(RuntimeError, match="Researcher judger requires a Brief"):
        researcher_facial_judge_batch([], brief=None)
    with pytest.raises(RuntimeError, match="Researcher judger requires a Brief"):
        researcher_full_judge(_make_candidate(), brief=None)
