"""Phase-0 shared-root regression: the judger must NOT swallow provider
budget/credit-exhaustion errors into a (retryable) ``JUDGMENT_FAILURE`` decision.

Background. Every judge in ``shared/judger.py`` wraps its LLM call in
``except Exception: return judgment_failure_decision(...)``. ``judgment_failure_decision``
correctly classifies a credit-exhaustion error as ``TERMINAL_ERROR /
api_budget_exhausted`` — but only to build a rationale string. The returned
``OpusDecision(decision='JUDGMENT_FAILURE')`` carries no terminal flag, and
``is_failure_decision`` + ``shared/execution/runtime.py`` treat
``JUDGMENT_FAILURE`` as retryable. So a dead/exhausted API key produced a
candidate marked retryable, the loop re-hit the dead key for every remaining
candidate for hours, and the run finalized ``status='completed'``. The
LinkedIn orchestrator's ``except Exception: if is_api_budget_exhausted_error: raise``
guard never fired because the judge RETURNED instead of RAISING.

The fix re-raises ``ApiBudgetExhaustedError`` at the top of every judge
except-handler (scoped to ``is_api_budget_exhausted_error`` only). These tests
pin BOTH halves of that contract:

  * a provider credit-exhaustion error RAISES ``ApiBudgetExhaustedError``
    (so the orchestrator guard fires and the run pauses);
  * a NON-budget runtime error STILL RETURNS a ``JUDGMENT_FAILURE`` decision
    (so the re-raise is scoped and ordinary transient failures are unchanged).

Unlike ``tests/test_linkedin_pipeline.py`` (which patches ``facial_judge``
*itself* to raise and so never exercises the real swallow), these patch the
real LLM clients — ``shared.judger.facial_llm`` and
``shared.judger.opus_llm_cached`` — so the actual except-handlers run.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import shared.judger as judger
from shared.failures import ApiBudgetExhaustedError, JUDGMENT_FAILURE
from shared.schemas import (
    CandidateProfileSummary,
    CandidateSnippet,
    Education,
    Experience,
    ExternalCandidateEvidence,
)


# The canonical Anthropic credit-exhaustion message. ``is_api_budget_exhausted_error``
# matches on the "credit balance is too low" / "purchase credits" substrings.
BUDGET_MESSAGE = (
    "Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to purchase credits."
)

# A transient error that must continue to absorb into a JUDGMENT_FAILURE decision.
NON_BUDGET_MESSAGE = "upstream connect error or disconnect/reset before headers"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_brief(*, has_v2: bool = True):
    """A MagicMock-shaped brief.

    The system-prompt assemblers are patched out in every test, so the brief's
    attribute surface is irrelevant to what we exercise (the except-handler).
    Only ``has_v2_schema`` is read by the judge before the LLM call.
    """

    brief = MagicMock()
    brief.has_v2_schema = has_v2
    brief._new_brief = MagicMock() if has_v2 else None
    return brief


def _make_snippet() -> CandidateSnippet:
    return CandidateSnippet(
        name="Jane Doe",
        headline="ML Researcher",
        current_title="Research Scientist",
        current_company="OpenLab",
        location="SF",
        education_snippet="PhD, MIT",
        profile_url="https://www.linkedin.com/in/janedoe",
        source_string_id=1,
        source_string_name="string",
        page=1,
        result_rank=1,
    )


def _make_summary() -> CandidateProfileSummary:
    return CandidateProfileSummary(
        name="Jane Doe",
        profile_url="https://www.linkedin.com/in/janedoe",
        headline="ML Researcher",
        experiences=[
            Experience(
                title="Research Scientist",
                company="OpenLab",
                start="2020",
                end="present",
                summary_bullets=["LLM evals"],
            )
        ],
        education=[Education(degree="PhD", school="MIT", field="ML")],
        skills_snippet=["python", "pytorch"],
    )


def _make_evidence() -> ExternalCandidateEvidence:
    return ExternalCandidateEvidence(
        trigger_reason="academic_context",
        identity_confidence=0.7,
        profile_facts_used_for_matching=["name=Jane Doe"],
        external_fact_blocks=[],
        external_inferences=[],
        unresolved_ambiguities=[],
        do_not_use_for_judgment=[],
        raw_provider_model="sonar-deep-research",
        normalizer_model="",
    )


def _stub_static_systems(stack):
    """Patch every system-prompt assembler the judges call to a static string.

    Keeps each test focused on the except-handler: with the assemblers stubbed,
    the only thing that can fail is the (patched) LLM call.
    """

    for name in (
        "assemble_facial_system",
        "assemble_full_evaluation_system",
        "_build_facial_system",
        "_build_full_system",
    ):
        stack.enter_context(patch.object(judger, name, return_value="SYSTEM"))


# ---------------------------------------------------------------------------
# Budget exhaustion RAISES — one case per facial_llm-backed judge entrypoint.
# ---------------------------------------------------------------------------

class TestBudgetExhaustionReraises:
    def test_facial_judge_v2_reraises(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            _stub_static_systems(stack)
            stack.enter_context(
                patch.object(judger, "facial_llm", side_effect=RuntimeError(BUDGET_MESSAGE))
            )
            with pytest.raises(ApiBudgetExhaustedError):
                judger.facial_judge(_make_snippet(), _make_brief(has_v2=True))

    def test_facial_judge_old_brief_reraises(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            _stub_static_systems(stack)
            stack.enter_context(
                patch.object(judger, "opus_llm_cached", side_effect=RuntimeError(BUDGET_MESSAGE))
            )
            with pytest.raises(ApiBudgetExhaustedError):
                judger.facial_judge(_make_snippet(), _make_brief(has_v2=False))

    def test_full_judge_v2_reraises(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            _stub_static_systems(stack)
            stack.enter_context(
                patch.object(judger, "opus_llm_cached", side_effect=RuntimeError(BUDGET_MESSAGE))
            )
            with pytest.raises(ApiBudgetExhaustedError):
                judger.full_judge(_make_summary(), _make_brief(has_v2=True))

    def test_full_judge_old_brief_reraises(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            _stub_static_systems(stack)
            stack.enter_context(
                patch.object(judger, "opus_llm_cached", side_effect=RuntimeError(BUDGET_MESSAGE))
            )
            with pytest.raises(ApiBudgetExhaustedError):
                judger.full_judge(_make_summary(), _make_brief(has_v2=False))

    def test_full_judge_with_external_evidence_v2_reraises(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            _stub_static_systems(stack)
            stack.enter_context(
                patch.object(judger, "opus_llm_cached", side_effect=RuntimeError(BUDGET_MESSAGE))
            )
            with pytest.raises(ApiBudgetExhaustedError):
                judger.full_judge_with_external_evidence(
                    _make_summary(), _make_evidence(), _make_brief(has_v2=True)
                )

    def test_full_judge_with_external_evidence_old_brief_reraises(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            _stub_static_systems(stack)
            stack.enter_context(
                patch.object(judger, "opus_llm_cached", side_effect=RuntimeError(BUDGET_MESSAGE))
            )
            with pytest.raises(ApiBudgetExhaustedError):
                judger.full_judge_with_external_evidence(
                    _make_summary(), _make_evidence(), _make_brief(has_v2=False)
                )

    def test_github_facial_judge_reraises(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            stack.enter_context(
                patch("github.judgment_templates.assemble_github_facial_system", return_value="SYSTEM")
            )
            stack.enter_context(
                patch.object(judger, "facial_llm", side_effect=RuntimeError(BUDGET_MESSAGE))
            )
            with pytest.raises(ApiBudgetExhaustedError):
                judger.github_facial_judge("portfolio text", _make_brief(has_v2=True))

    def test_github_full_judge_reraises(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            stack.enter_context(
                patch("github.judgment_templates.assemble_github_full_evaluation_system", return_value="SYSTEM")
            )
            stack.enter_context(
                patch.object(judger, "opus_llm_cached", side_effect=RuntimeError(BUDGET_MESSAGE))
            )
            with pytest.raises(ApiBudgetExhaustedError):
                judger.github_full_judge("evidence text", _make_brief(has_v2=True))

    def test_researcher_facial_judge_reraises(self):
        """Researcher facial path runs through ``researcher_facial_judge_batch``
        with a custom ``llm_caller`` that raises the budget error."""

        from contextlib import ExitStack

        with ExitStack() as stack:
            stack.enter_context(
                patch("researcher.judgment_templates.assemble_facial_system", return_value="SYSTEM")
            )
            stack.enter_context(
                patch("researcher.judgment_templates.render_facial_user_prompt", return_value="USER")
            )

            def _raise(*_args, **_kwargs):
                raise RuntimeError(BUDGET_MESSAGE)

            snippet = _make_researcher_snippet()
            with pytest.raises(ApiBudgetExhaustedError):
                judger.researcher_facial_judge_batch(
                    [snippet], _make_brief(has_v2=True), llm_caller=_raise
                )

    def test_researcher_full_judge_reraises(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            stack.enter_context(
                patch("researcher.judgment_templates.assemble_full_evaluation_system", return_value="SYSTEM")
            )
            stack.enter_context(
                patch("researcher.judgment_templates.render_full_user_prompt", return_value="USER")
            )

            def _raise(*_args, **_kwargs):
                raise RuntimeError(BUDGET_MESSAGE)

            with pytest.raises(ApiBudgetExhaustedError):
                judger.researcher_full_judge(
                    _make_researcher_snippet(), _make_brief(has_v2=True), llm_caller=_raise
                )

    def test_facial_judge_batch_fallback_reraises(self):
        """The batch path catches its own exception and falls back to
        per-snippet ``facial_judge``. Because ``facial_judge`` now re-raises on
        a budget error, the fallback must propagate it (not return failures)."""

        from contextlib import ExitStack

        with ExitStack() as stack:
            _stub_static_systems(stack)
            stack.enter_context(
                patch.object(judger, "assemble_facial_batch_system", return_value="SYSTEM")
            )
            stack.enter_context(
                patch.object(judger, "facial_llm", side_effect=RuntimeError(BUDGET_MESSAGE))
            )
            with pytest.raises(ApiBudgetExhaustedError):
                judger.facial_judge_batch([_make_snippet()], _make_brief(has_v2=True))


# ---------------------------------------------------------------------------
# Non-budget errors STILL ABSORB into JUDGMENT_FAILURE (scoping proof).
# ---------------------------------------------------------------------------

class TestNonBudgetErrorsStillAbsorb:
    def test_facial_judge_v2_non_budget_returns_failure(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            _stub_static_systems(stack)
            stack.enter_context(
                patch.object(judger, "facial_llm", side_effect=RuntimeError(NON_BUDGET_MESSAGE))
            )
            decision = judger.facial_judge(_make_snippet(), _make_brief(has_v2=True))
        assert decision.decision == JUDGMENT_FAILURE

    def test_full_judge_v2_non_budget_returns_failure(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            _stub_static_systems(stack)
            stack.enter_context(
                patch.object(judger, "opus_llm_cached", side_effect=RuntimeError(NON_BUDGET_MESSAGE))
            )
            decision = judger.full_judge(_make_summary(), _make_brief(has_v2=True))
        assert decision.decision == JUDGMENT_FAILURE

    def test_researcher_full_judge_non_budget_returns_failure(self):
        from contextlib import ExitStack

        with ExitStack() as stack:
            stack.enter_context(
                patch("researcher.judgment_templates.assemble_full_evaluation_system", return_value="SYSTEM")
            )
            stack.enter_context(
                patch("researcher.judgment_templates.render_full_user_prompt", return_value="USER")
            )

            def _raise(*_args, **_kwargs):
                raise RuntimeError(NON_BUDGET_MESSAGE)

            decision = judger.researcher_full_judge(
                _make_researcher_snippet(), _make_brief(has_v2=True), llm_caller=_raise
            )
        assert decision.decision == JUDGMENT_FAILURE


# ---------------------------------------------------------------------------
# Researcher snippet helper (researcher judges read .name / .profile_url plus
# fast-exit numeric floors; a MagicMock with high metrics survives fast-exit).
# ---------------------------------------------------------------------------

def _make_researcher_snippet():
    snippet = MagicMock()
    snippet.name = "Jane Doe"
    snippet.profile_url = "https://example.org/janedoe"
    # Generous metrics so the batch fast-exit gate does not short-circuit
    # before reaching the LLM call.
    snippet.h_index = 50
    snippet.papers_in_window = 50
    snippet.total_papers = 200
    return snippet
