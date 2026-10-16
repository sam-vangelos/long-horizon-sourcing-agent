"""Tests for the offline facial-gate experimental harness.

Hard rules pinned by these tests:

- No live LLM or network calls (callers inject ``facial_call``).
- ``--experiment`` is required; without it ``run`` exits with code 1.
- ``looser`` variant string-removes specific sentences from the baseline
  facial prompt and raises ``RuntimeError`` if the targeted substrings
  aren't present (so a future production-template churn cannot silently
  produce a no-op variant).
- ``FACIAL_BORDERLINE`` is only legal under ``variant='ternary'``; for
  any other variant it must parse as ``PARSE_FAILURE``.
- ``analyze_recovery`` is pure and the false-negative heuristic only
  fires when a profile_summary exists for the candidate (no evidence,
  no count).
- The harness imports nothing from the LinkedIn orchestrator / browser /
  acquisition / side_effects / work_units, github/, market_intelligence/,
  or shared/external_evidence/provider.py.
"""

from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from shared.brief_schema import (
    BiasControls,
    Brief as NewBrief,
    CapabilityArea,
    DepthDistinction,
    EmployerSignalRule,
    FacialCalibration,
    NonFitPattern,
)
from shared.schemas import (
    CandidateProfileSummary,
    CandidateSnippet,
    Education,
    Experience,
)

from tools.experiments import facial_gate_experiment as fge


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_new_brief(**overrides) -> NewBrief:
    """Real shared.brief_schema.Brief — the variant builder needs a v2 brief."""
    defaults = dict(
        role_title="Test Role",
        role_level="IC5",
        role_summary="Build the thing.",
        geography="United States",
        linkedin_project="proj-test",
        capability_areas=[
            CapabilityArea(
                name="Test Area",
                description="What the work looks like.",
                builder_signals=["builds X"],
                user_signals=[],
            )
        ],
        depth_distinction=DepthDistinction(
            builder_definition="Builds X end-to-end.",
            user_definition="Calls a hosted X API.",
            edge_case_guidance="Borderline cases default to user.",
        ),
        non_fit_patterns=[
            NonFitPattern(
                label="Adjacent thing",
                description="Looks like X but isn't.",
                why_not="Different stack.",
            )
        ],
        employer_signal_rules=[
            EmployerSignalRule(
                tier="neutral",
                employer_patterns=["Some Co"],
                evidence_required="hands-on X",
                save_on_employer_alone=False,
            )
        ],
        minimum_years_experience=3,
        minimum_bar_description="3y hands-on X.",
        facial_calibration=FacialCalibration(
            expected_yes_rate_low=0.25,
            expected_yes_rate_high=0.55,
            fast_exit_patterns=["pure ops"],
            trajectory_yes_patterns=["X engineering"],
            trajectory_ambiguous_patterns=["mixed"],
            trajectory_no_patterns=["pure frontend"],
        ),
        bias_controls=BiasControls(),
    )
    defaults.update(overrides)
    return NewBrief(**defaults)


@dataclass
class _OldBriefShim:
    """Tiny shim mimicking shared.brief_loader.Brief for the harness."""

    _new_brief: NewBrief
    role_title: str = "Test Role"


def _make_brief() -> _OldBriefShim:
    return _OldBriefShim(_new_brief=_make_new_brief())


def _make_snippet(
    *,
    name: str,
    profile_url: str,
    headline: str = "Engineer",
    current_title: str = "ML Engineer",
    current_company: str = "Acme",
    education_snippet: str = "BS CS",
    source_string_id: int = 1,
) -> CandidateSnippet:
    return CandidateSnippet(
        name=name,
        headline=headline,
        current_title=current_title,
        current_company=current_company,
        location="NYC",
        education_snippet=education_snippet,
        profile_url=profile_url,
        source_string_id=source_string_id,
        source_string_name="test-string",
        page=1,
        result_rank=1,
    )


def _make_facial_call(canned: dict[str, str]):
    """Build a deterministic facial_call keyed by candidate name."""

    def _call(system: str, user: str) -> str:
        for name, response in canned.items():
            if f"NAME: {name}" in user:
                return response
        return "DECISION: FACIAL_NO\nREASON: default"

    return _call


def _build_args(**kwargs):
    """Minimal argparse.Namespace builder for run() tests."""
    import argparse as _ap

    defaults = dict(
        experiment=True,
        variants=list(fge.VALID_VARIANTS),
        snippets=None,
        brief=None,
        profile_summaries=None,
        final_judgments=None,
        limit=50,
        json_out=None,
        max_candidates=50,
        ternary_policy="open_borderline",
    )
    defaults.update(kwargs)
    return _ap.Namespace(**defaults)


# ---------------------------------------------------------------------------
# load_snippets
# ---------------------------------------------------------------------------


class TestLoadSnippets:
    def test_happy_path_three_rows(self, tmp_path: Path):
        path = tmp_path / "snippets.jsonl"
        rows = [
            _make_snippet(name="Alice", profile_url="https://x/alice"),
            _make_snippet(name="Bob", profile_url="https://x/bob"),
            _make_snippet(name="Carol", profile_url="https://x/carol"),
        ]
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r.to_dict()) + "\n")
        loaded = fge.load_snippets(str(path), max_candidates=10)
        assert [s.name for s in loaded] == ["Alice", "Bob", "Carol"]

    def test_from_stdin_dash(self, monkeypatch):
        a = _make_snippet(name="Alice", profile_url="https://x/alice")
        b = _make_snippet(name="Bob", profile_url="https://x/bob")
        text = json.dumps(a.to_dict()) + "\n" + json.dumps(b.to_dict()) + "\n"
        monkeypatch.setattr("sys.stdin", io.StringIO(text))
        loaded = fge.load_snippets("-", max_candidates=10)
        assert [s.name for s in loaded] == ["Alice", "Bob"]

    def test_max_candidates_caps_input(self, tmp_path: Path):
        path = tmp_path / "snippets.jsonl"
        rows = [
            _make_snippet(name=f"P{i}", profile_url=f"https://x/{i}")
            for i in range(5)
        ]
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r.to_dict()) + "\n")
        loaded = fge.load_snippets(str(path), max_candidates=2)
        assert len(loaded) == 2
        assert [s.name for s in loaded] == ["P0", "P1"]

    def test_missing_file_returns_empty(self, tmp_path: Path):
        loaded = fge.load_snippets(
            str(tmp_path / "nope.jsonl"), max_candidates=10
        )
        assert loaded == []

    def test_skips_blank_and_malformed_lines(self, tmp_path: Path):
        path = tmp_path / "snippets.jsonl"
        good = _make_snippet(name="Alice", profile_url="https://x/alice")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n")
            fh.write("not-json\n")
            fh.write(json.dumps(good.to_dict()) + "\n")
        loaded = fge.load_snippets(str(path), max_candidates=10)
        assert [s.name for s in loaded] == ["Alice"]


# ---------------------------------------------------------------------------
# Index loaders
# ---------------------------------------------------------------------------


class TestProfileSummariesIndex:
    def test_empty_path_returns_empty(self):
        assert fge.load_profile_summaries_index(None) == {}
        assert fge.load_profile_summaries_index("") == {}

    def test_missing_path_returns_empty(self, tmp_path: Path):
        assert fge.load_profile_summaries_index(str(tmp_path / "x.jsonl")) == {}

    def test_indexes_by_profile_url(self, tmp_path: Path):
        path = tmp_path / "ps.jsonl"
        s = CandidateProfileSummary(
            name="Alice",
            profile_url="https://x/alice",
            headline="Eng",
            experiences=[Experience(title="Eng", company="Acme")],
            education=[Education(degree="PhD CS", school="MIT")],
            skills_snippet=["python"],
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(s.to_dict()) + "\n")
        index = fge.load_profile_summaries_index(str(path))
        assert "https://x/alice" in index
        assert index["https://x/alice"].name == "Alice"


class TestFinalJudgmentsIndex:
    def test_empty_returns_empty(self):
        assert fge.load_final_judgments_index(None) == {}

    def test_flat_decision_row(self, tmp_path: Path):
        path = tmp_path / "fj.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "profile_url": "https://x/alice",
                        "decision": "SAVE",
                        "candidate_name": "Alice",
                    }
                )
                + "\n"
            )
        idx = fge.load_final_judgments_index(str(path))
        assert "https://x/alice" in idx
        assert fge._extract_recorded_decision(idx["https://x/alice"]) == "SAVE"

    def test_nested_final_decision_shape(self, tmp_path: Path):
        path = tmp_path / "fj.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "profile_url": "https://x/bob",
                        "final_decision": {
                            "decision": "INFERENTIAL_SAVE",
                            "profile_url": "https://x/bob",
                        },
                    }
                )
                + "\n"
            )
        idx = fge.load_final_judgments_index(str(path))
        assert (
            fge._extract_recorded_decision(idx["https://x/bob"])
            == "INFERENTIAL_SAVE"
        )


# ---------------------------------------------------------------------------
# build_variant_system_prompt
# ---------------------------------------------------------------------------


class TestBuildVariantSystemPrompt:
    def test_baseline_contains_directly_connects_sentence(self):
        brief = _make_brief()
        prompt = fge.build_variant_system_prompt(brief, "baseline")
        assert "DIRECTLY connects" in prompt
        # Sanity: the canonical YES/NO output schema is intact.
        assert "FACIAL_YES" in prompt
        assert "FACIAL_NO" in prompt

    def test_looser_strips_target_sentences(self):
        brief = _make_brief()
        baseline = fge.build_variant_system_prompt(brief, "baseline")
        looser = fge.build_variant_system_prompt(brief, "looser")
        # The baseline must contain all three target substrings (otherwise
        # the looser builder would have raised).
        assert fge.LOOSER_STRIP_AMBIGUITY in baseline
        assert fge.LOOSER_STRIP_DO_NOT_OPEN in baseline
        assert fge.LOOSER_STRIP_DIRECTLY_BULLET in baseline
        # And the looser variant must NOT contain them.
        assert fge.LOOSER_STRIP_AMBIGUITY not in looser
        assert fge.LOOSER_STRIP_DO_NOT_OPEN not in looser
        assert fge.LOOSER_STRIP_DIRECTLY_BULLET not in looser
        # The replacement bullet is present.
        assert fge.LOOSER_REPLACEMENT_BULLET in looser

    def test_looser_preserves_non_fit_block(self):
        brief = _make_brief()
        looser = fge.build_variant_system_prompt(brief, "looser")
        # Non-fit block content from the brief fixture must survive.
        assert "Adjacent thing" in looser
        # fast_exit_block content from the brief fixture must survive too.
        assert "pure ops" in looser

    def test_ternary_includes_experimental_fence(self):
        brief = _make_brief()
        ternary = fge.build_variant_system_prompt(brief, "ternary")
        assert "=== EXPERIMENTAL TERNARY OUTPUT ===" in ternary
        assert "FACIAL_BORDERLINE" in ternary
        # Baseline content is intact under ternary (we appended, not mutated).
        assert "DIRECTLY connects" in ternary

    def test_looser_raises_when_target_substring_missing(self):
        brief = _make_brief()
        # Mock assemble_facial_system to return a prompt that doesn't contain
        # the target sentence — simulates a future production-template churn.
        with patch.object(
            fge,
            "build_variant_system_prompt",
            wraps=fge.build_variant_system_prompt,
        ):
            pass  # noop; we patch the source module instead.
        with patch(
            "linkedin.judgment_templates.assemble_facial_system",
            return_value="A totally unrelated prompt with FACIAL_YES and FACIAL_NO but nothing else.",
        ):
            with pytest.raises(RuntimeError, match="looser variant: substring"):
                fge.build_variant_system_prompt(brief, "looser")

    def test_unknown_variant_raises_value_error(self):
        brief = _make_brief()
        with pytest.raises(ValueError, match="unknown variant"):
            fge.build_variant_system_prompt(brief, "wat")


# ---------------------------------------------------------------------------
# parse_variant_response
# ---------------------------------------------------------------------------


class TestParseVariantResponse:
    def test_yes(self):
        d, _ = fge.parse_variant_response(
            "DECISION: FACIAL_YES\nREASON: strong", "baseline"
        )
        assert d == "FACIAL_YES"

    def test_no(self):
        d, _ = fge.parse_variant_response(
            "DECISION: FACIAL_NO\nREASON: nope", "baseline"
        )
        assert d == "FACIAL_NO"

    def test_borderline_legal_under_ternary(self):
        d, r = fge.parse_variant_response(
            "DECISION: FACIAL_BORDERLINE\nREASON: ambiguous", "ternary"
        )
        assert d == "FACIAL_BORDERLINE"
        assert r == "ambiguous"

    def test_borderline_illegal_under_baseline(self):
        d, _ = fge.parse_variant_response(
            "DECISION: FACIAL_BORDERLINE\nREASON: ambiguous", "baseline"
        )
        assert d == "PARSE_FAILURE"

    def test_borderline_illegal_under_looser(self):
        d, _ = fge.parse_variant_response(
            "DECISION: FACIAL_BORDERLINE\nREASON: ambiguous", "looser"
        )
        assert d == "PARSE_FAILURE"

    def test_garbage_is_parse_failure(self):
        d, _ = fge.parse_variant_response("the model went off-script", "ternary")
        assert d == "PARSE_FAILURE"

    def test_empty_is_parse_failure(self):
        d, _ = fge.parse_variant_response("", "baseline")
        assert d == "PARSE_FAILURE"

    def test_decision_without_explicit_label(self):
        # If only "FACIAL_YES" appears in the body, fall back.
        d, _ = fge.parse_variant_response(
            "I think FACIAL_YES is appropriate.", "baseline"
        )
        assert d == "FACIAL_YES"


# ---------------------------------------------------------------------------
# run_variant_against_snippets
# ---------------------------------------------------------------------------


class TestRunVariantAgainstSnippets:
    def _four_snippets(self):
        return [
            _make_snippet(name="Alice", profile_url="https://x/alice"),
            _make_snippet(name="Bob", profile_url="https://x/bob"),
            _make_snippet(name="Carol", profile_url="https://x/carol"),
            _make_snippet(name="Dan", profile_url="https://x/dan"),
        ]

    def _canned(self):
        # 2 YES, 1 NO, 1 BORDERLINE.
        return {
            "Alice": "DECISION: FACIAL_YES\nREASON: clear",
            "Bob": "DECISION: FACIAL_YES\nREASON: clear",
            "Carol": "DECISION: FACIAL_NO\nREASON: outside scope",
            "Dan": "DECISION: FACIAL_BORDERLINE\nREASON: ambiguous",
        }

    def test_baseline_counters(self):
        brief = _make_brief()
        result = fge.run_variant_against_snippets(
            snippets=self._four_snippets(),
            brief=brief,
            variant="baseline",
            facial_call=_make_facial_call(self._canned()),
            ternary_policy="open_borderline",
        )
        assert result.total_snippets == 4
        assert result.facial_yes == 2
        assert result.facial_no == 1
        # BORDERLINE under baseline parses as PARSE_FAILURE.
        assert result.facial_borderline == 0
        assert result.parse_failures == 1
        # reach_full_eval == facial_yes for binary variants.
        assert result.reach_full_eval == 2

    def test_ternary_open_borderline_reaches_full_eval(self):
        brief = _make_brief()
        result = fge.run_variant_against_snippets(
            snippets=self._four_snippets(),
            brief=brief,
            variant="ternary",
            facial_call=_make_facial_call(self._canned()),
            ternary_policy="open_borderline",
        )
        assert result.facial_yes == 2
        assert result.facial_no == 1
        assert result.facial_borderline == 1
        assert result.parse_failures == 0
        # 2 YES + 1 BORDERLINE = 3 reach full eval.
        assert result.reach_full_eval == 3

    def test_ternary_skip_borderline_excludes_borderline(self):
        brief = _make_brief()
        result = fge.run_variant_against_snippets(
            snippets=self._four_snippets(),
            brief=brief,
            variant="ternary",
            facial_call=_make_facial_call(self._canned()),
            ternary_policy="skip_borderline",
        )
        assert result.facial_borderline == 1
        # Borderline does NOT count toward reach_full_eval.
        assert result.reach_full_eval == 2

    def test_token_proxy_and_latency_are_recorded(self):
        brief = _make_brief()
        result = fge.run_variant_against_snippets(
            snippets=self._four_snippets(),
            brief=brief,
            variant="baseline",
            facial_call=_make_facial_call(self._canned()),
            ternary_policy="open_borderline",
        )
        assert result.input_token_proxy_total > 0
        assert result.output_token_proxy_total > 0
        assert result.latency_total_seconds >= 0.0
        # cost_per_reached_full_eval_proxy is input_total / reach_full_eval (2).
        assert result.cost_per_reached_full_eval_proxy == pytest.approx(
            result.input_token_proxy_total / 2
        )


# ---------------------------------------------------------------------------
# analyze_recovery
# ---------------------------------------------------------------------------


def _build_results_for_recovery():
    """4 candidates: A, B, C, D.

    baseline:  A=YES B=YES C=NO  D=NO
    looser:    A=YES B=YES C=YES D=YES   (looser opens everyone)

    final_judgments:
      A: SAVE        (baseline opened, looser opened — baseline floor)
      B: REJECT      (both opened, recorded run rejected)
      C: SAVE        (baseline did NOT open; looser would have)
      D: REJECT
    """
    brief = _make_brief()
    snippets = [
        _make_snippet(name="A", profile_url="https://x/a"),
        _make_snippet(name="B", profile_url="https://x/b"),
        _make_snippet(name="C", profile_url="https://x/c"),
        _make_snippet(name="D", profile_url="https://x/d"),
    ]
    base_canned = {
        "A": "DECISION: FACIAL_YES\nREASON: ok",
        "B": "DECISION: FACIAL_YES\nREASON: ok",
        "C": "DECISION: FACIAL_NO\nREASON: skip",
        "D": "DECISION: FACIAL_NO\nREASON: skip",
    }
    looser_canned = {
        "A": "DECISION: FACIAL_YES\nREASON: ok",
        "B": "DECISION: FACIAL_YES\nREASON: ok",
        "C": "DECISION: FACIAL_YES\nREASON: opened by looser",
        "D": "DECISION: FACIAL_YES\nREASON: opened by looser",
    }
    base_result = fge.run_variant_against_snippets(
        snippets=snippets,
        brief=brief,
        variant="baseline",
        facial_call=_make_facial_call(base_canned),
    )
    looser_result = fge.run_variant_against_snippets(
        snippets=snippets,
        brief=brief,
        variant="looser",
        facial_call=_make_facial_call(looser_canned),
    )
    final_judgments_index = {
        "https://x/a": {"profile_url": "https://x/a", "decision": "SAVE"},
        "https://x/b": {"profile_url": "https://x/b", "decision": "REJECT"},
        "https://x/c": {"profile_url": "https://x/c", "decision": "SAVE"},
        "https://x/d": {"profile_url": "https://x/d", "decision": "REJECT"},
    }
    return brief, base_result, looser_result, final_judgments_index


class TestAnalyzeRecovery:
    def test_recovery_math(self):
        brief, base, looser, fj = _build_results_for_recovery()
        out = fge.analyze_recovery(
            baseline_result=base,
            variant_result=looser,
            final_judgments_index=fj,
            profile_summaries_index={},
            brief=brief,
        )
        # baseline opened A and B (both YES) → 1 of those (A) saved → floor = 1.
        assert out["baseline_saves_recovered"] == 1
        # looser opened all 4. Of those, A and C saved → 2.
        assert out["variant_saves_recovered"] == 2
        # variant_only_recovered_saves: candidates the variant opened that
        # baseline did NOT open AND who saved — that's C. 1.
        assert out["variant_only_recovered_saves"] == 1
        assert out["recovery_evidence_available"] is True

    def test_agreement_disagreement_buckets(self):
        brief, base, looser, fj = _build_results_for_recovery()
        out = fge.analyze_recovery(
            baseline_result=base,
            variant_result=looser,
            final_judgments_index=fj,
            profile_summaries_index={},
            brief=brief,
        )
        # A and B both YES on both → 2 agreements.
        assert out["agreement"] == 2
        # C and D: baseline NO, looser YES → both fall in
        # baseline_no_variant_yes.
        assert out["baseline_no_variant_yes"] == 2
        assert out["disagreement"] == 2

    def test_likely_false_negatives_with_phd_summary(self):
        brief = _make_brief()
        # Variant says NO to Alice; profile_summary has a PhD entry → counted.
        snippets = [
            _make_snippet(name="Alice", profile_url="https://x/alice"),
            _make_snippet(name="Bob", profile_url="https://x/bob"),
        ]
        base_canned = {
            "Alice": "DECISION: FACIAL_YES\nREASON: ok",
            "Bob": "DECISION: FACIAL_YES\nREASON: ok",
        }
        # Hypothetical "tighter" variant — NO on both.
        tighter_canned = {
            "Alice": "DECISION: FACIAL_NO\nREASON: tightened out",
            "Bob": "DECISION: FACIAL_NO\nREASON: tightened out",
        }
        base = fge.run_variant_against_snippets(
            snippets=snippets,
            brief=brief,
            variant="baseline",
            facial_call=_make_facial_call(base_canned),
        )
        tighter = fge.run_variant_against_snippets(
            snippets=snippets,
            brief=brief,
            variant="baseline",  # we just reuse the baseline prompt builder
            facial_call=_make_facial_call(tighter_canned),
        )
        # Profile summary index: Alice has a PhD; Bob has no entry.
        ps_index = {
            "https://x/alice": CandidateProfileSummary(
                name="Alice",
                profile_url="https://x/alice",
                headline="PhD",
                experiences=[
                    Experience(
                        title="Researcher",
                        company="Univ",
                        summary_bullets=["did research", "published"],
                    )
                ],
                education=[
                    Education(degree="PhD Computer Science", school="MIT")
                ],
                skills_snippet=[],
            ),
            # NOTE: deliberately no entry for Bob to test the
            # "no profile summary → not counted" rule.
        }
        out = fge.analyze_recovery(
            baseline_result=base,
            variant_result=tighter,
            final_judgments_index={},
            profile_summaries_index=ps_index,
            brief=brief,
        )
        # Alice has a PhD entry → counted in the bucket.
        # Bob has no profile summary → NOT counted (no evidence to gate on).
        assert out["likely_false_negatives_under_variant"] == 1
        assert out["likely_false_negative_urls"] == ["https://x/alice"]


# ---------------------------------------------------------------------------
# run() — end-to-end with mocked facial_call
# ---------------------------------------------------------------------------


class TestRunEndToEnd:
    def _write_v2_brief(self, tmp_path: Path) -> Path:
        """Write a minimal V2-shaped brief that load_brief can ingest."""
        brief = {
            "role_title": "Test Role",
            "role_level": "IC5",
            "role_summary": "Build the thing.",
            "geography": "United States",
            "linkedin_project": "proj-test",
            "minimum_years_experience": 3,
            "minimum_bar_description": "3y hands-on X.",
            "capability_areas": [
                {
                    "name": "Test Area",
                    "description": "What the work looks like.",
                    "builder_signals": ["builds X"],
                    "user_signals": [],
                }
            ],
            "depth_distinction": {
                "builder_definition": "Builds X end-to-end.",
                "user_definition": "Calls a hosted X API.",
                "edge_case_guidance": "Borderline cases default to user.",
            },
            "non_fit_patterns": [
                {
                    "label": "Adjacent thing",
                    "description": "Looks like X but isn't.",
                    "why_not": "Different stack.",
                }
            ],
            "employer_signal_rules": [
                {
                    "tier": "neutral",
                    "employer_patterns": ["Some Co"],
                    "evidence_required": "hands-on X",
                    "save_on_employer_alone": False,
                }
            ],
            "facial_calibration": {
                "expected_yes_rate_low": 0.25,
                "expected_yes_rate_high": 0.55,
                "fast_exit_patterns": ["pure ops"],
                "trajectory_yes_patterns": ["X engineering"],
                "trajectory_ambiguous_patterns": ["mixed"],
                "trajectory_no_patterns": ["pure frontend"],
            },
        }
        p = tmp_path / "brief.json"
        p.write_text(json.dumps(brief))
        return p

    def _write_snippets(self, tmp_path: Path) -> Path:
        rows = [
            _make_snippet(name="Alice", profile_url="https://x/alice"),
            _make_snippet(name="Bob", profile_url="https://x/bob"),
            _make_snippet(name="Carol", profile_url="https://x/carol"),
            _make_snippet(name="Dan", profile_url="https://x/dan"),
        ]
        p = tmp_path / "snippets.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r.to_dict()) + "\n")
        return p

    def _write_summaries(self, tmp_path: Path) -> Path:
        rows = [
            CandidateProfileSummary(
                name="Carol",
                profile_url="https://x/carol",
                headline="PhD researcher",
                experiences=[
                    Experience(title="PhD student", company="Univ")
                ],
                education=[Education(degree="PhD CS", school="MIT")],
                skills_snippet=[],
            ),
        ]
        p = tmp_path / "summaries.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r.to_dict()) + "\n")
        return p

    def _write_final_judgments(self, tmp_path: Path) -> Path:
        rows = [
            {"profile_url": "https://x/alice", "decision": "SAVE"},
            {"profile_url": "https://x/bob", "decision": "REJECT"},
        ]
        p = tmp_path / "final.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return p

    def test_no_experiment_flag_exits_one(self, capsys):
        args = _build_args(experiment=False, snippets="x", brief="y")
        rc = fge.run(args, facial_call=_make_facial_call({}))
        assert rc == 1
        err = capsys.readouterr().err
        assert "experiment harness" in err

    def test_no_snippets_resolved_exits_one(self, tmp_path: Path, capsys):
        brief_path = self._write_v2_brief(tmp_path)
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        args = _build_args(snippets=str(empty), brief=str(brief_path))
        rc = fge.run(args, facial_call=_make_facial_call({}))
        assert rc == 1
        err = capsys.readouterr().err
        assert "no snippets resolved" in err

    def test_all_three_variants_end_to_end(self, tmp_path: Path, capsys):
        brief_path = self._write_v2_brief(tmp_path)
        snippets_path = self._write_snippets(tmp_path)
        summaries_path = self._write_summaries(tmp_path)
        final_path = self._write_final_judgments(tmp_path)
        json_out = tmp_path / "summary.json"

        canned = {
            "Alice": "DECISION: FACIAL_YES\nREASON: ok",
            "Bob": "DECISION: FACIAL_YES\nREASON: ok",
            "Carol": "DECISION: FACIAL_NO\nREASON: skip",
            "Dan": "DECISION: FACIAL_BORDERLINE\nREASON: ambiguous",
        }
        args = _build_args(
            snippets=str(snippets_path),
            brief=str(brief_path),
            profile_summaries=str(summaries_path),
            final_judgments=str(final_path),
            json_out=str(json_out),
        )
        rc = fge.run(args, facial_call=_make_facial_call(canned))
        assert rc == 0
        out = capsys.readouterr().out
        # All three variants present in stdout.
        assert "variant: baseline" in out
        assert "variant: looser" in out
        assert "variant: ternary" in out
        # Pairwise comparisons present.
        assert "baseline vs looser" in out
        assert "baseline vs ternary" in out
        # JSON-out file matches the in-memory summary shape.
        assert json_out.exists()
        loaded = json.loads(json_out.read_text())
        assert set(loaded["variants"].keys()) == {"baseline", "looser", "ternary"}
        assert "looser" in loaded["comparisons"]
        assert "ternary" in loaded["comparisons"]
        # Recovery available because final-judgments was provided.
        assert (
            loaded["comparisons"]["looser"]["recovery_evidence_available"]
            is True
        )

    def test_ternary_only_skips_pairwise(self, tmp_path: Path, capsys):
        brief_path = self._write_v2_brief(tmp_path)
        snippets_path = self._write_snippets(tmp_path)
        canned = {
            "Alice": "DECISION: FACIAL_YES\nREASON: ok",
            "Bob": "DECISION: FACIAL_YES\nREASON: ok",
            "Carol": "DECISION: FACIAL_NO\nREASON: skip",
            "Dan": "DECISION: FACIAL_BORDERLINE\nREASON: ambiguous",
        }
        args = _build_args(
            snippets=str(snippets_path),
            brief=str(brief_path),
            variants=["ternary"],
        )
        rc = fge.run(args, facial_call=_make_facial_call(canned))
        assert rc == 0
        out = capsys.readouterr().out
        assert "variant: ternary" in out
        assert "variant: baseline" not in out
        # Pairwise comparisons section is present but explicitly notes no baseline.
        assert "Pairwise Comparisons" in out
        assert "skipped" in out
        assert "baseline variant was not run" in out


# ---------------------------------------------------------------------------
# Forbidden imports — keep the harness narrow.
# ---------------------------------------------------------------------------


class TestImportNarrowness:
    def test_harness_does_not_import_orchestrator_or_browser(self):
        import tools.experiments.facial_gate_experiment as harness
        text = Path(harness.__file__).read_text()
        for forbidden in (
            "from linkedin.orchestrator",
            "from linkedin.browser",
            "from linkedin.acquisition",
            "from linkedin.side_effects",
            "from linkedin.work_units",
            "from github.",
            "from market_intelligence",
            "from shared.external_evidence.provider",
        ):
            assert forbidden not in text, (
                f"harness must not import from forbidden module: {forbidden}"
            )
