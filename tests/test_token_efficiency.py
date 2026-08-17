"""Tests for token efficiency optimization (prompt caching, batch facial, model downgrade).

Run with: python -m pytest tests/test_token_efficiency.py -v
"""

from unittest.mock import MagicMock, patch, call
import json

from shared.schemas import CandidateSnippet, OpusDecision


def _make_snippet(**kwargs) -> CandidateSnippet:
    defaults = {
        "name": "Test Person",
        "headline": "ML Engineer",
        "current_title": "ML Engineer",
        "current_company": "Acme Corp",
        "location": "San Francisco",
        "education_snippet": "BS CS Stanford",
        "profile_url": "/talent/profile/test123",
        "source_string_id": 1,
        "source_string_name": "test",
        "page": 1,
        "result_rank": 1,
    }
    defaults.update(kwargs)
    return CandidateSnippet(**defaults)


# ---------------------------------------------------------------------------
# Phase 1: Prompt Caching — opus_llm_cached passes cache_control
# ---------------------------------------------------------------------------

def _mock_anthropic_client():
    """Create a mock Anthropic client with standard success response."""
    mock_msg = MagicMock()
    mock_msg.stop_reason = "end_turn"
    mock_msg.content = [MagicMock(text='{"answer": "yes"}')]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    mock_module = MagicMock()
    mock_module.Anthropic.return_value = mock_client
    return mock_module, mock_client


class TestOpusLlmCached:
    def test_passes_cache_control_annotation(self):
        """opus_llm_cached sends system as content blocks with cache_control."""
        mock_module, mock_client = _mock_anthropic_client()

        import shared.llm_clients as llm_mod
        orig_config = llm_mod.config
        try:
            llm_mod.config = MagicMock(ANTHROPIC_API_KEY="k", OPUS_MODEL_NAME="claude-opus-4-6")
            with patch.dict("sys.modules", {"anthropic": mock_module}):
                llm_mod.opus_llm_cached("system prompt", "user prompt")

            call_kwargs = mock_client.messages.create.call_args
            system_arg = call_kwargs.kwargs["system"]
            assert isinstance(system_arg, list)
            assert system_arg[0]["type"] == "text"
            assert system_arg[0]["text"] == "system prompt"
            assert system_arg[0]["cache_control"] == {"type": "ephemeral"}
        finally:
            llm_mod.config = orig_config

    def test_opus_llm_still_passes_string_system(self):
        """opus_llm (non-cached) still passes system as a plain string."""
        mock_module, mock_client = _mock_anthropic_client()

        import shared.llm_clients as llm_mod
        orig_config = llm_mod.config
        try:
            llm_mod.config = MagicMock(ANTHROPIC_API_KEY="k", OPUS_MODEL_NAME="claude-opus-4-6")
            with patch.dict("sys.modules", {"anthropic": mock_module}):
                llm_mod.opus_llm("system prompt", "user prompt")

            call_kwargs = mock_client.messages.create.call_args
            system_arg = call_kwargs.kwargs["system"]
            assert isinstance(system_arg, str)
            assert system_arg == "system prompt"
        finally:
            llm_mod.config = orig_config


# ---------------------------------------------------------------------------
# Phase 1: Split assembly functions produce correct static prefix
# ---------------------------------------------------------------------------

class TestSplitAssembly:
    def _make_brief(self):
        """Create a minimal mock brief for template testing."""
        brief = MagicMock()
        brief.role_title = "ML Engineer"
        brief.role_level = "IC4"
        brief.role_summary = "Build ML systems"
        brief.fast_exit_block.return_value = "- Wrong domain entirely"
        brief.trajectory_yes_block.return_value = "- ML research positions"
        brief.trajectory_ambiguous_block.return_value = "- Mixed ML/non-ML"
        brief.trajectory_no_block.return_value = "- Pure frontend"
        brief.non_fit_block.return_value = "- Management consulting"
        brief.capability_area_names.return_value = ["Data Curation", "RL"]
        brief.trajectory_yes_compact.return_value = "ML research"
        brief.trajectory_ambiguous_compact.return_value = "Mixed"
        brief.trajectory_no_compact.return_value = "Frontend"
        brief.non_fit_compact.return_value = "Consulting"
        brief.capability_area_names_inline.return_value = "Data Curation, RL"
        # Full eval fields
        brief.minimum_years_experience = 5
        brief.minimum_bar_description = "Hands-on ML"
        brief.capability_area_block.return_value = "1. Data Curation"
        brief.depth_block.return_value = "Builder vs User"
        brief.non_fit_override_rule_block.return_value = "Override rule"
        brief.employer_signal_block.return_value = "Employer rules"
        brief.inferential_save_block.return_value = "Inferential rules"
        brief.discriminating_skills_examples.return_value = "custom kernels, distributed training"
        brief.seniority_calibration_block.return_value = ""
        brief.executive_builder_block.return_value = ""
        brief.decision_matrix_block.return_value = "Decision matrix"
        brief.post_evaluation_safety_net.return_value = ""
        brief.post_save_modifiers_block.return_value = ""
        brief.calibration_block.return_value = ""
        brief.instructions_block.return_value = ""
        brief.capability_area_stack_rank_guidance.return_value = ""
        return brief

    def test_facial_system_contains_no_candidate_data(self):
        from linkedin.judgment_templates import assemble_facial_system
        brief = self._make_brief()
        system = assemble_facial_system(brief)

        assert "ML Engineer" in system
        assert "FACIAL_YES" in system
        assert "FACIAL_NO" in system
        # Should NOT contain actual candidate data — just a placeholder
        assert "[provided in user message]" in system

    def test_full_evaluation_system_contains_no_candidate_data(self):
        from linkedin.judgment_templates import assemble_full_evaluation_system
        brief = self._make_brief()
        system = assemble_full_evaluation_system(brief)

        assert "ML Engineer" in system
        assert "SAVE" in system
        assert "REJECT" in system
        assert "[provided in user message]" in system

    def test_facial_batch_system_contains_no_candidate_data(self):
        from linkedin.judgment_templates import assemble_facial_batch_system
        brief = self._make_brief()
        system = assemble_facial_batch_system(brief)

        assert "ML Engineer" in system
        assert "FACIAL_YES" in system
        assert "[provided in user message]" in system

    def test_github_facial_system_default_portfolio_patterns_are_domain_neutral(self):
        from github.judgment_templates import assemble_github_facial_system
        brief = self._make_brief()
        brief.github_portfolio_yes_patterns = None
        brief.github_fast_exit_patterns = None
        brief.github_portfolio_ambiguous_patterns = None
        brief.github_portfolio_no_patterns = None
        system = assemble_github_facial_system(brief)

        assert "comparably significant projects in its domain" in system
        assert "huggingface" not in system.lower()
        assert "frontier" not in system.lower()
        for term in (
            "rlhf",
            "reward-model",
            "llm-evaluation",
            "machine learning",
            "artificial intelligence",
        ):
            assert term not in system.lower()

    def test_github_facial_system_contains_no_candidate_data(self):
        from github.judgment_templates import assemble_github_facial_system
        brief = self._make_brief()
        system = assemble_github_facial_system(brief)

        assert "ML Engineer" in system
        assert "FACIAL_YES" in system
        assert "[provided in user message]" in system

    def test_github_facial_batch_system_contains_no_candidate_data(self):
        from github.judgment_templates import assemble_github_facial_batch_system
        brief = self._make_brief()
        system = assemble_github_facial_batch_system(brief)

        assert "ML Engineer" in system
        assert "FACIAL_YES" in system
        assert "[provided in user message]" in system

    def test_github_full_evaluation_system_contains_no_candidate_data(self):
        from github.judgment_templates import assemble_github_full_evaluation_system
        brief = self._make_brief()
        system = assemble_github_full_evaluation_system(brief)

        assert "ML Engineer" in system
        assert "SAVE" in system
        assert "[provided in user message]" in system
        assert "TARGET PROJECT CONTRIBUTIONS" in system
        assert "other recognized high-signal repositories in the candidate's domain" in system
        assert "huggingface" not in system.lower()

    # -----------------------------------------------------------------
    # P5.1 + P5.3: confluence (synthesize-don't-checklist) restored to
    # the V2 templates, plus a brief-agnostic education-weighting slot.
    # Both must be framed as evidence FOR clearing the bar, never as a
    # hedge for uncertain candidates (strict-ambiguity doctrine intact).
    # -----------------------------------------------------------------

    def test_facial_system_contains_confluence_and_education_synthesis(self):
        from linkedin.judgment_templates import assemble_facial_system
        brief = self._make_brief()
        system = assemble_facial_system(brief)

        assert "SYNTHESIZE, DON'T CHECKLIST" in system
        assert (
            "Education (degree level, field, institution) is a supporting signal"
            in system
        )
        assert "never sufficient on its own and never required for a YES" in system
        # Doctrine guard: confluence/education must not read as a hedge, and
        # the pre-existing strict-ambiguity line must survive untouched.
        assert "when uncertain" not in system.lower()
        assert "lean facial_yes" not in system.lower()
        assert "Ambiguity favors NO" in system

    def test_facial_batch_system_contains_confluence_and_education_synthesis(self):
        from linkedin.judgment_templates import assemble_facial_batch_system
        brief = self._make_brief()
        system = assemble_facial_batch_system(brief)

        assert "SYNTHESIZE:" in system
        assert (
            "Education (degree, field, institution) is a supporting signal only"
            in system
        )
        assert "never sufficient alone, never required" in system
        assert "Ambiguity favors NO" in system

    def test_full_evaluation_system_contains_career_read_and_education_synthesis(self):
        # P1 evaluator redesign: the full-eval synthesis paragraph became the
        # whole-career read (plans/glm-evaluator-operating-philosophy.md §C.3).
        # The facial synthesis pin above is unchanged — facial templates kept
        # the confluence paragraph.
        from linkedin.judgment_templates import assemble_full_evaluation_system
        brief = self._make_brief()
        system = assemble_full_evaluation_system(brief)

        assert "SYNTHESIZE — READ A CAREER, NOT A DOCUMENT" in system
        assert "the whole-career read wins" in system
        assert (
            "Education (degree level, field, institution) remains one input to the synthesis"
            in system
        )
        assert "never sufficient alone and never required to reach SAVE" in system
        assert "when uncertain" not in system.lower()


# ---------------------------------------------------------------------------
# Phase 2: Batch facial response parser
# ---------------------------------------------------------------------------

class TestParseFacialBatchResponse:
    def test_well_formed_response(self):
        from linkedin.judgment_templates import parse_facial_batch_response
        raw = (
            "[1] FACIAL_YES | Strong ML trajectory at DeepMind\n"
            "[2] FACIAL_NO | Pure frontend developer\n"
            "[3] FACIAL_YES | PhD in RL from Stanford\n"
        )
        results = parse_facial_batch_response(raw, 3)
        assert len(results) == 3
        assert results[0].decision == "FACIAL_YES"
        assert "DeepMind" in results[0].reason
        assert results[1].decision == "FACIAL_NO"
        assert results[2].decision == "FACIAL_YES"

    def test_missing_entry_returns_parse_failure(self):
        from linkedin.judgment_templates import parse_facial_batch_response
        raw = (
            "[1] FACIAL_YES | Good candidate\n"
            "[3] FACIAL_NO | Not a fit\n"
        )
        results = parse_facial_batch_response(raw, 3)
        assert len(results) == 3
        assert results[0].decision == "FACIAL_YES"
        assert results[1].decision == "PARSE_FAILURE"
        assert "missing" in results[1].reason
        assert results[2].decision == "FACIAL_NO"

    def test_malformed_line_ignored(self):
        from linkedin.judgment_templates import parse_facial_batch_response
        raw = (
            "[1] FACIAL_YES | Good\n"
            "This is not a valid line\n"
            "[2] FACIAL_NO | Bad\n"
        )
        results = parse_facial_batch_response(raw, 2)
        assert len(results) == 2
        assert results[0].decision == "FACIAL_YES"
        assert results[1].decision == "FACIAL_NO"

    def test_empty_response(self):
        from linkedin.judgment_templates import parse_facial_batch_response
        results = parse_facial_batch_response("", 3)
        assert len(results) == 3
        assert all(r.decision == "PARSE_FAILURE" for r in results)

    def test_case_insensitive(self):
        from linkedin.judgment_templates import parse_facial_batch_response
        raw = "[1] facial_yes | Good fit\n"
        results = parse_facial_batch_response(raw, 1)
        assert results[0].decision == "FACIAL_YES"


# ---------------------------------------------------------------------------
# Phase 2: facial_judge_batch returns correct decisions
# ---------------------------------------------------------------------------

class TestFacialJudgeBatch:
    def test_batch_returns_correct_count(self):
        """facial_judge_batch returns one OpusDecision per snippet."""
        snippets = [
            _make_snippet(name="Alice", profile_url="/alice"),
            _make_snippet(name="Bob", profile_url="/bob"),
            _make_snippet(name="Carol", profile_url="/carol"),
        ]

        mock_brief = MagicMock()
        mock_brief.has_v2_schema = True
        mock_brief._new_brief = MagicMock()

        batch_response = (
            "[1] FACIAL_YES | ML trajectory\n"
            "[2] FACIAL_NO | Wrong domain\n"
            "[3] FACIAL_YES | RL researcher\n"
        )

        with patch("shared.judger.facial_llm", return_value=batch_response), \
             patch("shared.judger.assemble_facial_batch_system", return_value="system"):
            from shared.judger import facial_judge_batch
            decisions = facial_judge_batch(snippets, mock_brief)

        assert len(decisions) == 3
        assert decisions[0].decision == "FACIAL_YES"
        assert decisions[0].candidate_name == "Alice"
        assert decisions[1].decision == "FACIAL_NO"
        assert decisions[1].candidate_name == "Bob"
        assert decisions[2].decision == "FACIAL_YES"
        assert decisions[2].candidate_name == "Carol"

    def test_batch_falls_back_to_sequential_on_failure(self):
        """If batch call fails, falls back to individual facial_judge calls."""
        snippets = [_make_snippet(name="Alice", profile_url="/alice")]

        mock_brief = MagicMock()
        mock_brief.has_v2_schema = True
        mock_brief._new_brief = MagicMock()

        with patch("shared.judger.facial_llm", side_effect=RuntimeError("API down")), \
             patch("shared.judger.assemble_facial_batch_system", return_value="system"), \
             patch("shared.judger.facial_judge") as mock_sequential:
            mock_sequential.return_value = OpusDecision(
                stage="facial", decision="FACIAL_YES", path="none",
                confidence=1.0, rationale="sequential fallback",
                candidate_name="Alice", profile_url="/alice",
            )

            from shared.judger import facial_judge_batch
            decisions = facial_judge_batch(snippets, mock_brief)

        assert len(decisions) == 1
        assert decisions[0].decision == "FACIAL_YES"
        mock_sequential.assert_called_once()

    def test_count_mismatch_rejudges_whole_batch_sequentially(self):
        """A partial batch (fewer valid verdicts than snippets) re-judges the WHOLE batch.

        Phase-0 mis-attribution fix: this test previously asserted keep-prefix /
        retry-tail-only (Alice keeps batch [1], only Bob retried). That behavior
        is exactly the verdict mis-attribution defect — when the model drops a
        candidate and renumbers survivors 1..K, the "kept" prefix slot holds a
        renumbered survivor's verdict pinned to the wrong snippet, and only the
        trailing gap is ever cross-checked. Because a dense-prefix + tail-gap
        response is wire-indistinguishable from a legitimate drop-the-last-one
        response, the safe contract is fail-loud: any count mismatch re-judges
        the entire batch sequentially so every verdict re-attaches to the right
        person. The well-formed in-order path (one [N] per candidate) is
        unaffected — see test_batch_returns_correct_count.
        """
        snippets = [
            _make_snippet(name="Alice", profile_url="/alice"),
            _make_snippet(name="Bob", profile_url="/bob"),
        ]

        mock_brief = MagicMock()
        mock_brief.has_v2_schema = True
        mock_brief._new_brief = MagicMock()

        batch_response = "[1] FACIAL_YES | ML trajectory\n"

        def _sequential(snippet, brief, prompt_prefix="", lane_context=None):
            del brief, prompt_prefix, lane_context
            return OpusDecision(
                stage="facial", decision="FACIAL_NO", path="none",
                confidence=1.0, rationale=f"sequential:{snippet.name}",
                candidate_name=snippet.name, profile_url=snippet.profile_url,
            )

        with patch("shared.judger.facial_llm", return_value=batch_response), \
             patch("shared.judger.assemble_facial_batch_system", return_value="system"), \
             patch("shared.judger.facial_judge", side_effect=_sequential) as mock_sequential:
            from shared.judger import facial_judge_batch
            decisions = facial_judge_batch(snippets, mock_brief)

        # Both snippets re-judged sequentially; neither retains a batch verdict.
        assert len(decisions) == 2
        by_name = {d.candidate_name: d for d in decisions}
        assert by_name["Alice"].rationale == "sequential:Alice"
        assert by_name["Bob"].rationale == "sequential:Bob"
        assert mock_sequential.call_count == 2

    def test_old_brief_uses_sequential(self):
        """Old briefs (no V2 schema) fall back to sequential facial_judge."""
        snippets = [_make_snippet(name="Alice", profile_url="/alice")]

        mock_brief = MagicMock()
        mock_brief.has_v2_schema = False

        with patch("shared.judger.facial_judge") as mock_sequential:
            mock_sequential.return_value = OpusDecision(
                stage="facial", decision="FACIAL_NO", path="none",
                confidence=1.0, rationale="old brief path",
                candidate_name="Alice", profile_url="/alice",
            )

            from shared.judger import facial_judge_batch
            decisions = facial_judge_batch(snippets, mock_brief)

        assert len(decisions) == 1
        assert decisions[0].decision == "FACIAL_NO"


class TestGitHubFacialJudgeBatch:
    def test_github_batch_fallback_preserves_metadata(self):
        portfolio_texts = [
            ("Alice Example", "https://github.com/alice", "portfolio 1"),
        ]

        mock_brief = MagicMock()
        mock_brief.has_v2_schema = True
        mock_brief._new_brief = MagicMock()

        with patch("shared.judger.facial_llm", side_effect=RuntimeError("API down")), \
             patch("shared.judger.github_facial_judge") as mock_single:
            mock_single.return_value = OpusDecision(
                stage="facial", decision="FACIAL_YES", path="none",
                confidence=1.0, rationale="fallback", candidate_name="", profile_url="",
            )

            from shared.judger import github_facial_judge_batch
            decisions = github_facial_judge_batch(portfolio_texts, mock_brief)

        assert len(decisions) == 1
        assert decisions[0].candidate_name == "Alice Example"
        assert decisions[0].profile_url == "https://github.com/alice"
        assert decisions[0].decision == "FACIAL_YES"


# ---------------------------------------------------------------------------
# Phase 3: FACIAL_MODEL_NAME defaults to OPUS_MODEL_NAME
# ---------------------------------------------------------------------------

class TestFacialModelConfig:
    def test_unset_facial_model_falls_back_to_opus(self, monkeypatch):
        import shared.config as config

        monkeypatch.delenv("FACIAL_MODEL_NAME", raising=False)
        assert (
            config._optional("FACIAL_MODEL_NAME", config.OPUS_MODEL_NAME)
            == config.OPUS_MODEL_NAME
        )

    def test_facial_llm_uses_facial_model_name(self):
        """facial_llm should use config.FACIAL_MODEL_NAME, not OPUS_MODEL_NAME."""
        mock_module, mock_client = _mock_anthropic_client()
        mock_client.messages.create.return_value.content[0].text = "DECISION: FACIAL_YES\nREASON: good"

        import shared.llm_clients as llm_mod
        orig_config = llm_mod.config
        try:
            llm_mod.config = MagicMock(
                ANTHROPIC_API_KEY="k",
                FACIAL_MODEL_NAME="claude-sonnet-4-6",
            )
            with patch.dict("sys.modules", {"anthropic": mock_module}):
                llm_mod.facial_llm("system", "user", expect_json=False)

            call_kwargs = mock_client.messages.create.call_args
            assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"
        finally:
            llm_mod.config = orig_config

    def test_facial_llm_lower_default_max_tokens(self):
        """facial_llm should default to max_tokens=2048."""
        mock_module, mock_client = _mock_anthropic_client()

        import shared.llm_clients as llm_mod
        orig_config = llm_mod.config
        try:
            llm_mod.config = MagicMock(
                ANTHROPIC_API_KEY="k",
                FACIAL_MODEL_NAME="claude-opus-4-6",
            )
            with patch.dict("sys.modules", {"anthropic": mock_module}):
                llm_mod.facial_llm("system", "user")

            call_kwargs = mock_client.messages.create.call_args
            assert call_kwargs.kwargs["max_tokens"] == 2048
        finally:
            llm_mod.config = orig_config


# ---------------------------------------------------------------------------
# Step B (slice 13): FACIAL_BORDERLINE parser + flag-gated prompt selection
# ---------------------------------------------------------------------------
# Pins the parser-widening + ternary-template behavior introduced in Step B
# of the FACIAL_BORDERLINE promotion plan. The parser recognizes BORDERLINE
# unconditionally; the prompt assembler picks the ternary template only
# when ``LINKEDIN_FACIAL_BORDERLINE_ENABLED`` is True. Production behavior
# under flag-off is byte-identical to pre-Step-B.
# ---------------------------------------------------------------------------


class TestFacialBorderlineParser:
    def test_parse_facial_response_recognizes_borderline_in_decision_line(self):
        from linkedin.judgment_templates import parse_facial_response
        raw = "DECISION: FACIAL_BORDERLINE\nREASON: snippet cannot resolve"
        result = parse_facial_response(raw)
        assert result.decision == "FACIAL_BORDERLINE"
        assert "cannot resolve" in result.reason

    def test_parse_facial_response_borderline_wins_over_yes_substring_check(self):
        """Order-matters fix: BORDERLINE detection runs before YES/NO substring."""
        from linkedin.judgment_templates import parse_facial_response
        raw = "DECISION: FACIAL_BORDERLINE\nREASON: yes-flavored prose mentioning yes"
        result = parse_facial_response(raw)
        assert result.decision == "FACIAL_BORDERLINE"

    def test_parse_facial_batch_response_recognizes_borderline(self):
        from linkedin.judgment_templates import parse_facial_batch_response
        raw = (
            "[1] FACIAL_BORDERLINE | snippet cannot resolve\n"
            "[2] FACIAL_YES | strong ML trajectory\n"
            "[3] FACIAL_NO | pure frontend\n"
        )
        results = parse_facial_batch_response(raw, 3)
        assert len(results) == 3
        assert results[0].decision == "FACIAL_BORDERLINE"
        assert results[1].decision == "FACIAL_YES"
        assert results[2].decision == "FACIAL_NO"

    def test_parse_facial_response_unknown_decision_still_parse_failure(self):
        """Parser must not become permissive — unknown classes still fail."""
        from linkedin.judgment_templates import parse_facial_response
        raw = "DECISION: FACIAL_MAYBE\nREASON: model went off-script"
        result = parse_facial_response(raw)
        assert result.decision == "PARSE_FAILURE"


class TestFacialBorderlinePromptSelection:
    """Flag-gated selection of binary vs ternary facial templates.

    These tests monkey-patch ``shared.config.LINKEDIN_FACIAL_BORDERLINE_ENABLED``
    rather than setting the env-var, because the flag is read once at import
    time. ``linkedin.judgment_templates`` accesses it lazily via
    ``_config.LINKEDIN_FACIAL_BORDERLINE_ENABLED`` so an attribute monkey-patch
    is observed by ``assemble_facial_system`` / ``assemble_facial_batch_system``.
    """

    def _make_brief(self):
        brief = MagicMock()
        brief.role_title = "ML Engineer"
        brief.role_level = "IC4"
        brief.role_summary = "Build ML systems"
        brief.fast_exit_block.return_value = "- Wrong domain entirely"
        brief.trajectory_yes_block.return_value = "- ML research positions"
        brief.trajectory_ambiguous_block.return_value = "- Mixed ML/non-ML"
        brief.trajectory_no_block.return_value = "- Pure frontend"
        brief.non_fit_block.return_value = "- Management consulting"
        brief.capability_area_names.return_value = ["Data Curation", "RL"]
        brief.trajectory_yes_compact.return_value = "ML research"
        brief.trajectory_ambiguous_compact.return_value = "Mixed"
        brief.trajectory_no_compact.return_value = "Frontend"
        brief.non_fit_compact.return_value = "Consulting"
        brief.capability_area_names_inline.return_value = "Data Curation, RL"
        return brief

    def test_assemble_facial_system_uses_binary_template_under_flag_off(self, monkeypatch):
        import shared.config as cfg
        from linkedin.judgment_templates import assemble_facial_system
        monkeypatch.setattr(cfg, "LINKEDIN_FACIAL_BORDERLINE_ENABLED", False)
        result = assemble_facial_system(self._make_brief())
        assert "Ambiguity favors NO" in result
        assert "DECISION: FACIAL_YES or FACIAL_NO" in result
        assert "FACIAL_BORDERLINE" not in result

    def test_assemble_facial_system_uses_ternary_template_under_flag_on(self, monkeypatch):
        import shared.config as cfg
        from linkedin.judgment_templates import assemble_facial_system
        monkeypatch.setattr(cfg, "LINKEDIN_FACIAL_BORDERLINE_ENABLED", True)
        result = assemble_facial_system(self._make_brief())
        assert "Ambiguity favors NO" not in result
        assert "Do NOT open a profile just to" not in result
        assert "FACIAL_BORDERLINE" in result
        assert "DECISION: FACIAL_YES | FACIAL_BORDERLINE | FACIAL_NO" in result

    def test_assemble_facial_batch_system_flag_off_unchanged(self, monkeypatch):
        import shared.config as cfg
        from linkedin.judgment_templates import assemble_facial_batch_system
        monkeypatch.setattr(cfg, "LINKEDIN_FACIAL_BORDERLINE_ENABLED", False)
        result = assemble_facial_batch_system(self._make_brief())
        assert "Ambiguity favors NO" in result
        assert "[candidate_number] FACIAL_YES or FACIAL_NO" in result
        assert "FACIAL_BORDERLINE" not in result

    def test_assemble_facial_batch_system_flag_on_ternary(self, monkeypatch):
        import shared.config as cfg
        from linkedin.judgment_templates import assemble_facial_batch_system
        monkeypatch.setattr(cfg, "LINKEDIN_FACIAL_BORDERLINE_ENABLED", True)
        result = assemble_facial_batch_system(self._make_brief())
        assert "Ambiguity favors NO" not in result
        assert "FACIAL_BORDERLINE" in result
        assert "[candidate_number] FACIAL_YES | FACIAL_BORDERLINE | FACIAL_NO" in result

    def test_ternary_synthesize_block_routes_ambiguity_to_borderline_not_no(self, monkeypatch):
        """P5.1 hardening (Opus review finding): the SYNTHESIZE block imported
        into the ternary templates must not tell the judge that ambiguity
        favors NO — the ternary's own doctrine routes brief-listed ambiguous
        patterns to FACIAL_BORDERLINE ("Do NOT collapse ambiguity to NO").
        The pre-existing flag-on guards checked the capitalized "Ambiguity
        favors NO" sentence only, which let a lowercase "ambiguity still
        favors NO" clause slip into the batch-ternary prompt. Casing-proof
        check on both ternary assemblies.
        """
        import shared.config as cfg
        from linkedin.judgment_templates import (
            assemble_facial_system,
            assemble_facial_batch_system,
        )
        monkeypatch.setattr(cfg, "LINKEDIN_FACIAL_BORDERLINE_ENABLED", True)
        for assemble in (assemble_facial_system, assemble_facial_batch_system):
            system = assemble(self._make_brief())
            assert "favors no" not in system.lower(), (
                f"{assemble.__name__}: ternary prompt tells the judge ambiguity "
                "favors NO, contradicting its own BORDERLINE routing"
            )
        batch = assemble_facial_batch_system(self._make_brief())
        assert "ambiguity favors BORDERLINE, never YES" in batch

    def test_ternary_templates_within_token_efficiency_budget(self):
        """Ternary template must be at most 1.20× binary template length.

        The auditor flagged token-proxy growth as a cost concern. This pin
        catches future prompt drift that would balloon the cached prefix.
        """
        from linkedin.judgment_templates import (
            FACIAL_TRIAGE_TEMPLATE,
            FACIAL_TRIAGE_TEMPLATE_TERNARY,
            FACIAL_TRIAGE_TEMPLATE_BATCH,
            FACIAL_TRIAGE_TEMPLATE_BATCH_TERNARY,
        )
        single_ratio = len(FACIAL_TRIAGE_TEMPLATE_TERNARY) / len(FACIAL_TRIAGE_TEMPLATE)
        batch_ratio = len(FACIAL_TRIAGE_TEMPLATE_BATCH_TERNARY) / len(FACIAL_TRIAGE_TEMPLATE_BATCH)
        assert single_ratio <= 1.20, (
            f"Ternary single template too large: ratio={single_ratio:.4f} "
            f"(binary={len(FACIAL_TRIAGE_TEMPLATE)}, ternary={len(FACIAL_TRIAGE_TEMPLATE_TERNARY)})"
        )
        assert batch_ratio <= 1.20, (
            f"Ternary batch template too large: ratio={batch_ratio:.4f} "
            f"(binary={len(FACIAL_TRIAGE_TEMPLATE_BATCH)}, ternary={len(FACIAL_TRIAGE_TEMPLATE_BATCH_TERNARY)})"
        )
