"""Adversarial completeness tests for the judge dead-credit re-raise fix.

Companion to ``tests/test_judger_budget_exhaustion.py`` (the implementer's
suite). That suite covers the per-judge except-handlers plus the LinkedIn
``facial_judge_batch`` fallback. This file pins the paths it leaves uncovered
and the contract boundaries an adversarial reviewer must confirm are intact:

  1. ``github_facial_judge_batch`` fallback — the GitHub analog of the
     LinkedIn batch fallback. Its ``except Exception`` (judger.py:1504) does
     NOT carry a ``_reraise_if_budget_exhausted`` guard; correctness relies
     ENTIRELY on the per-item ``github_facial_judge`` re-raising. Same class
     as the LinkedIn batch path; the implementer tested LinkedIn, not GitHub.

  2. Batch-fallback scoping — a NON-budget error inside the batch path must
     STILL collapse to a ``JUDGMENT_FAILURE`` decision list (not raise),
     mirroring the per-judge scoping contract for the batch fallback.

  3. The external-evidence contract boundary. ``full_judge_with_external_evidence``
     is a JUDGE and re-raises (proved in the implementer's suite). The
     external-evidence PROVIDER (``shared/external_evidence/provider.py``) is a
     SEPARATE function that MUST NOT raise / pause. These tests pin both halves
     so a future refactor that conflates them (e.g. importing the judge's
     re-raise into the provider path) is caught.

All tests are written to FAIL against the pre-fix ``shared/judger.py`` for the
re-raise cases (verified by git-stashing the fix), and to PASS for the
scoping / provider-contract cases (which characterize pre-existing behavior the
fix must not regress).
"""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest

import shared.judger as judger
from shared.failures import (
    ApiBudgetExhaustedError,
    JUDGMENT_FAILURE,
    is_api_budget_exhausted_error,
)
from shared.schemas import CandidateSnippet


BUDGET_MESSAGE = (
    "Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to purchase credits."
)
NON_BUDGET_MESSAGE = "upstream connect error or disconnect/reset before headers"


def _make_brief(*, has_v2: bool = True):
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


# ---------------------------------------------------------------------------
# 1. github_facial_judge_batch fallback — uncovered population member.
# ---------------------------------------------------------------------------

class TestGithubBatchFallbackReraises:
    def test_github_facial_judge_batch_fallback_reraises(self):
        """``github_facial_judge_batch`` catches its own batch-call exception
        (judger.py:1504) and falls back to per-item ``github_facial_judge``.
        Because ``github_facial_judge`` now re-raises on a budget error, the
        fallback MUST propagate it rather than return failure decisions.

        Pre-fix this returns a list of JUDGMENT_FAILURE decisions (no raise);
        post-fix it raises ApiBudgetExhaustedError. Same defect class as the
        LinkedIn batch fallback, which the implementer's suite covers."""

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "github.judgment_templates.assemble_github_facial_batch_system",
                    return_value="SYSTEM",
                )
            )
            # The per-item fallback path calls github_facial_judge, which
            # assembles the single-facial system prompt.
            stack.enter_context(
                patch(
                    "github.judgment_templates.assemble_github_facial_system",
                    return_value="SYSTEM",
                )
            )
            # facial_llm raises the budget error for BOTH the batch call and
            # the per-item fallback call.
            stack.enter_context(
                patch.object(
                    judger, "facial_llm", side_effect=RuntimeError(BUDGET_MESSAGE)
                )
            )
            with pytest.raises(ApiBudgetExhaustedError):
                judger.github_facial_judge_batch(
                    [("janedoe", "https://github.com/janedoe", "portfolio text")],
                    _make_brief(has_v2=True),
                )


# ---------------------------------------------------------------------------
# 2. Batch-fallback scoping — non-budget errors STILL absorb.
# ---------------------------------------------------------------------------

class TestBatchFallbackScoping:
    def test_linkedin_batch_fallback_non_budget_returns_failures(self):
        """A NON-budget error in the LinkedIn batch path must collapse to a
        JUDGMENT_FAILURE decision list via the per-snippet fallback — NOT raise.

        This pins the scoping half for the batch path: the implementer's suite
        proves the batch fallback RAISES on a budget error, but does not prove
        it still ABSORBS a transient error. A reviewer must confirm the
        re-raise is scoped at the batch boundary too."""

        with ExitStack() as stack:
            for name in (
                "assemble_facial_system",
                "_build_facial_system",
            ):
                stack.enter_context(patch.object(judger, name, return_value="SYSTEM"))
            stack.enter_context(
                patch.object(judger, "assemble_facial_batch_system", return_value="SYSTEM")
            )
            stack.enter_context(
                patch.object(
                    judger, "facial_llm", side_effect=RuntimeError(NON_BUDGET_MESSAGE)
                )
            )
            decisions = judger.facial_judge_batch(
                [_make_snippet()], _make_brief(has_v2=True)
            )
        assert len(decisions) == 1
        assert decisions[0].decision == JUDGMENT_FAILURE

    def test_github_batch_fallback_non_budget_returns_failures(self):
        """GitHub batch analog of the scoping check above."""

        with ExitStack() as stack:
            stack.enter_context(
                patch(
                    "github.judgment_templates.assemble_github_facial_batch_system",
                    return_value="SYSTEM",
                )
            )
            stack.enter_context(
                patch(
                    "github.judgment_templates.assemble_github_facial_system",
                    return_value="SYSTEM",
                )
            )
            stack.enter_context(
                patch.object(
                    judger, "facial_llm", side_effect=RuntimeError(NON_BUDGET_MESSAGE)
                )
            )
            decisions = judger.github_facial_judge_batch(
                [("janedoe", "https://github.com/janedoe", "portfolio text")],
                _make_brief(has_v2=True),
            )
        assert len(decisions) == 1
        assert decisions[0].decision == JUDGMENT_FAILURE


# ---------------------------------------------------------------------------
# 3. External-evidence contract boundary (judge raises; provider must not).
# ---------------------------------------------------------------------------

class TestExternalEvidenceProviderContractIntact:
    """The PROVIDER must not raise / pause on a budget error; the JUDGE must.

    These are deliberately separate functions in separate modules. The fix
    touched only the judge. A reviewer must confirm the provider's no-pause
    contract is intact and that the judge's re-raise has not leaked into it.
    """

    def test_provider_does_not_raise_on_quota_error(self):
        """``fetch_external_candidate_evidence`` must return an
        ``ExternalEvidenceFailure(reason='quota_exhausted')`` — NEVER raise
        ApiBudgetExhaustedError — even when the underlying client raises a
        canonical credit-exhaustion error. Candidate-level external evidence
        MUST NOT pause a LinkedIn run."""

        from shared.external_evidence.provider import (
            fetch_external_candidate_evidence,
        )
        from shared.schemas import (
            CandidateProfileSummary,
            Education,
            Experience,
            ExternalEvidenceFailure,
            TriggerDecision,
        )

        summary = CandidateProfileSummary(
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
            skills_snippet=["python"],
        )
        brief = MagicMock()
        brief.role_summary = "ML role"
        brief.role_title = "ML Researcher"
        trigger = TriggerDecision(
            should_run=True, reason="academic_context", skip_reason="", signals={}
        )

        with ExitStack() as stack:
            # Enable the provider path past its config gates.
            stack.enter_context(
                patch("shared.external_evidence.provider.config", MagicMock(
                    PERPLEXITY_API_KEY="sk-test",
                    LINKEDIN_EXTERNAL_EVIDENCE_ENABLED=True,
                    LINKEDIN_EXTERNAL_EVIDENCE_MODEL="sonar",
                    LINKEDIN_EXTERNAL_EVIDENCE_TIMEOUT_SECONDS=90.0,
                    LINKEDIN_EXTERNAL_EVIDENCE_MAX_OUTPUT_TOKENS=4096,
                    LINKEDIN_EXTERNAL_EVIDENCE_PERPLEXITY_PRESET="",
                ))
            )

            # Force the OpenAI client constructor to raise the budget error so
            # the provider's own except-handler (provider.py:414) runs.
            fake_openai_module = MagicMock()
            fake_openai_module.OpenAI.side_effect = RuntimeError(BUDGET_MESSAGE)
            stack.enter_context(
                patch.dict("sys.modules", {"openai": fake_openai_module})
            )

            result = fetch_external_candidate_evidence(
                summary=summary, brief=brief, trigger=trigger
            )

        assert isinstance(result, ExternalEvidenceFailure), (
            "provider must return a typed failure, not raise"
        )
        assert result.reason == "quota_exhausted", (
            f"budget error must map to quota_exhausted, got {result.reason!r}"
        )

    def test_provider_module_does_not_reference_budget_reraise(self):
        """Static contract guard: the provider module must NOT import or call
        the judge's budget re-raise helpers. If a future refactor wires
        ``ApiBudgetExhaustedError`` / ``is_api_budget_exhausted_error`` into the
        provider, candidate-level evidence would start pausing the run — the
        exact conflation the fix's docstring warns against.

        Checks the module's executable surface (imported / bound names and the
        AST of executable statements), NOT the docstring — which legitimately
        names ``ApiBudgetExhaustedError`` to explain why it is never raised."""

        import ast
        import inspect

        import shared.external_evidence.provider as provider_mod

        forbidden = {
            "ApiBudgetExhaustedError",
            "is_api_budget_exhausted_error",
            "_reraise_if_budget_exhausted",
        }

        # 1. None of the forbidden names are bound in the module namespace
        #    (i.e. not imported / not defined).
        bound = set(vars(provider_mod))
        assert not (forbidden & bound), (
            f"provider must not import/define budget-pause symbols: "
            f"{forbidden & bound}"
        )

        # 2. None of the forbidden names appear as a Name/Attribute reference
        #    anywhere in the module's executable AST (docstrings are Constant
        #    nodes and are not Name references, so they are correctly ignored).
        tree = ast.parse(inspect.getsource(provider_mod))
        referenced: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, ast.alias):
                referenced.add(node.asname or node.name)
        assert not (forbidden & referenced), (
            f"provider must not reference budget-pause symbols in code: "
            f"{forbidden & referenced}"
        )


# ---------------------------------------------------------------------------
# 4. Classifier scoping — the pause trigger must not over-match.
# ---------------------------------------------------------------------------

class TestBudgetClassifierScoping:
    def test_ordinary_transient_message_is_not_budget(self):
        """A transient connect-reset error must NOT be classified as budget
        exhaustion, or every transient blip would pause the run instead of
        retrying. Pins the lower bound of the re-raise scope."""

        assert is_api_budget_exhausted_error(RuntimeError(NON_BUDGET_MESSAGE)) is False

    def test_rate_limit_message_is_not_budget(self):
        """A 429 / rate-limit error is RECOVERABLE, not budget-terminal. It
        must not trip the pause path."""

        assert is_api_budget_exhausted_error(
            RuntimeError("429 Too Many Requests: rate limit exceeded")
        ) is False

    def test_canonical_credit_message_is_budget(self):
        assert is_api_budget_exhausted_error(RuntimeError(BUDGET_MESSAGE)) is True
