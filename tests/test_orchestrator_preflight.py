"""Orchestrator-level preflight tests (P9.2, P9.3).

P9.2: a v2 preflight failure must retry once, then abort the run with a
typed error — never silently fall back to a legacy evaluation regime.
(The legacy ``shared/preflight.py`` module — whose hardcoded ML archetype
reinstated the exact permissiveness failure v2 was built to fix — was
itself deleted in P10 once this fix left it with zero production
callers; ``_run_preflight_v2`` never imports it.)

P9.3: a successfully generated v2 brief is stamped with
``provenance: {generated_by, reviewed: False, generated_at}`` before it
is written and reloaded, and the ``format_for_review`` rendering is
printed at generation time so an operator watching the run sees the
eval criteria before candidates are judged against them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_pipeline(output_dir: str, tmp_path: Path):
    with patch("linkedin.orchestrator.load_brief") as mock_brief, \
         patch("linkedin.orchestrator.init_judger"), \
         patch("linkedin.orchestrator.LinkedInBrowser"):
        brief = MagicMock()
        brief.id = "test-brief"
        brief.linkedin_project_id = "test-project"
        brief.linkedin_project = ""
        brief.has_v2_schema = False
        brief.employer_blacklist = []
        brief.kit_url = ""
        brief.jd_text = "We need a senior backend engineer."
        brief.permanent_filters = {}
        brief.needs_preflight = MagicMock(return_value=False)
        mock_brief.return_value = brief

        brief_path = tmp_path / "brief.json"
        brief_path.write_text(json.dumps({"id": "test-brief"}))

        from linkedin.orchestrator import Pipeline

        pipeline = Pipeline(brief_path=str(brief_path), output_dir=output_dir)
        pipeline._original_brief_obj = brief
        return pipeline


def _valid_preflight_dict() -> dict:
    """A lint-clean generated brief (P4: the lint now gates go-live, so the
    canned success-path response must actually pass it — named capability
    areas, non-empty non-fit patterns, a derived band with rationale, no
    disposition language, hiring company blacklisted and in no tier)."""
    return {
        "role_title": "Senior Backend Engineer",
        "role_level": "Senior",
        "role_summary": "Owns payment infra.",
        "hiring_company": "Acme Fintech",
        "employer_blacklist": ["Acme Fintech"],
        "engagement_context": {
            "hiring_company": "Acme Fintech",
            "engagement_description": "A search for a hands-on payments owner.",
            "talent_bar_statement": "Critical payment-rail ownership clears the bar.",
            "selectivity_posture": "selective",
        },
        "capability_areas": [
            {
                "name": "Payments infrastructure",
                "description": "Owns payment rails end to end.",
                "builder_signals": ["designed ledger systems"],
                "user_signals": ["integrated a payments API"],
                "key_terms": ["idempotency"],
            }
        ],
        "depth_distinction": {
            "builder_definition": "Builds the payment rails themselves.",
            "user_definition": "Calls a payments API someone else built.",
            "edge_case_guidance": "Look for ownership of failure handling.",
        },
        "non_fit_patterns": [
            {
                "label": "Frontend checkout dev",
                "description": "Builds checkout UI only.",
                "why_not": "Never touches ledger or settlement.",
                "examples": ["Checkout page redesign"],
            }
        ],
        "employer_signal_rules": [
            {
                "tier": "general_tech",
                "employer_patterns": ["Stripe"],
                "evidence_required": "ledger ownership",
                "save_on_employer_alone": False,
            }
        ],
        "minimum_years_experience": 5,
        "minimum_bar_description": "5+ years owning payment infra.",
        "facial_calibration": {
            "expected_yes_rate_low": 0.15,
            "expected_yes_rate_high": 0.35,
            "yes_rate_rationale": "Moderate-density senior backend pool in this metro.",
            "fast_exit_patterns": ["entire career in graphic design"],
            "trajectory_yes_patterns": ["ledger systems engineer at a payments company"],
            "trajectory_ambiguous_patterns": ["backend engineer, domain unclear from snippet"],
            "trajectory_no_patterns": ["entire career in marketing"],
        },
        "domain_lane_hints": [
            {"lane": "payments_processors", "patterns": ["stripe", "adyen", "payments"]}
        ],
        "market_density": "moderate",
    }


def _valid_preflight_json() -> str:
    return json.dumps(_valid_preflight_dict())


# ---------------------------------------------------------------------------
# P9.3 — success path stamps provenance + prints the review rendering
# ---------------------------------------------------------------------------


def test_preflight_v2_success_stamps_provenance_and_prints_review(
    tmp_path, monkeypatch, capsys
):
    pipeline = _make_pipeline(str(tmp_path), tmp_path)

    monkeypatch.setattr(
        "shared.llm_clients.opus_llm", lambda *a, **k: _valid_preflight_json()
    )

    fake_reloaded_brief = MagicMock()
    fake_reloaded_brief.has_v2_schema = False
    mock_load_v2 = MagicMock(return_value=fake_reloaded_brief)
    monkeypatch.setattr("shared.brief_loader._load_v2_brief", mock_load_v2)

    with patch("linkedin.orchestrator.init_judger"):
        pipeline._run_preflight_v2()

    # The brief JSON written to disk carries honest, never-reviewed provenance.
    generated_path = pipeline.output_dir / "preflight_v2_brief.json"
    assert generated_path.exists()
    written = json.loads(generated_path.read_text())
    provenance = written["provenance"]
    assert provenance["generated_by"] == "preflight_v2"
    assert provenance["reviewed"] is False
    assert provenance["generated_at"]  # non-empty timestamp string
    assert written["linkedin_project_id"] == "test-project"
    assert written["source_config"]["linkedin"]["project_id"] == "test-project"

    # The same provenance-bearing dict is what gets reloaded as the live brief
    # — the future P9.5/report-header detector reads this off brief_obj.raw.
    mock_load_v2.assert_called_once()
    (reloaded_arg,) = mock_load_v2.call_args.args
    assert reloaded_arg["provenance"]["reviewed"] is False
    assert reloaded_arg["linkedin_project_id"] == "test-project"
    assert reloaded_arg["source_config"]["linkedin"]["project_id"] == "test-project"

    # format_for_review was printed to the console at generation time —
    # format_for_review had zero callers before this fix.
    captured = capsys.readouterr()
    assert "PREFLIGHT REVIEW" in captured.out


# ---------------------------------------------------------------------------
# P9.2(a) — retry-once-then-abort, never a silent regime swap
# ---------------------------------------------------------------------------


def test_preflight_v2_retries_once_then_succeeds(tmp_path, monkeypatch, capsys):
    pipeline = _make_pipeline(str(tmp_path), tmp_path)

    calls = {"n": 0}
    contexts: list[dict] = []

    def _flaky(*args, **kwargs):
        calls["n"] += 1
        context = kwargs["usage_context"]
        contexts.append(dict(context))
        assert "logical_call_id" not in context
        context["logical_call_id"] = f"llm-attempt-{calls['n']}"
        if calls["n"] == 1:
            raise RuntimeError("transient provider error")
        return _valid_preflight_json()

    monkeypatch.setattr("shared.llm_clients.opus_llm", _flaky)

    fake_reloaded_brief = MagicMock()
    fake_reloaded_brief.has_v2_schema = False
    monkeypatch.setattr(
        "shared.brief_loader._load_v2_brief", MagicMock(return_value=fake_reloaded_brief)
    )

    with patch("linkedin.orchestrator.init_judger"):
        pipeline._run_preflight_v2()  # must not raise

    assert calls["n"] == 2
    assert contexts[0]["parent_logical_call_id"] == contexts[1]["parent_logical_call_id"]
    assert "logical_call_id" not in contexts[0]
    assert "logical_call_id" not in contexts[1]

    captured = capsys.readouterr()
    assert "retrying once" in captured.err


def test_preflight_v2_typed_load_failure_does_not_spend_outer_retry(
    tmp_path, monkeypatch
):
    pipeline = _make_pipeline(str(tmp_path), tmp_path)
    calls = {"n": 0}

    def _success(*_args, **_kwargs):
        calls["n"] += 1
        return _valid_preflight_json()

    monkeypatch.setattr("shared.llm_clients.opus_llm", _success)
    loader = MagicMock(side_effect=ValueError("local typed-loader failure"))
    monkeypatch.setattr("shared.brief_loader._load_v2_brief", loader)

    with patch("linkedin.orchestrator.init_judger"):
        with pytest.raises(ValueError, match="local typed-loader failure"):
            pipeline._run_preflight_v2()

    assert calls["n"] == 1
    loader.assert_called_once()


def test_preflight_v2_aborts_after_two_failures_never_falls_back_to_legacy(
    tmp_path, monkeypatch, capsys
):
    pipeline = _make_pipeline(str(tmp_path), tmp_path)
    original_brief_obj = pipeline.brief_obj

    calls = {"n": 0}

    def _always_fails(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError(f"provider outage #{calls['n']}")

    monkeypatch.setattr("shared.llm_clients.opus_llm", _always_fails)

    mock_load_v2 = MagicMock()
    monkeypatch.setattr("shared.brief_loader._load_v2_brief", mock_load_v2)

    from shared.preflight_v2 import PreflightRegimeError

    with patch("linkedin.orchestrator.init_judger"):
        with pytest.raises(PreflightRegimeError):
            pipeline._run_preflight_v2()

    # Exactly one retry — not a retry loop, not a single silent try.
    assert calls["n"] == 2

    # Never fell back to a legacy evaluation regime (shared/preflight.py,
    # which used to be the fallback target, was deleted in P10).
    assert not (pipeline.output_dir / "preflight_output.json").exists()

    # No brief swap happened — the run is dead, not quietly running on a
    # different evaluation regime than the operator asked for.
    mock_load_v2.assert_not_called()
    assert pipeline.brief_obj is original_brief_obj


# ---------------------------------------------------------------------------
# P4 (sourcing-rigor-hardening) — the generated-brief lint gates go-live
# through the same retry→abort path as a parse failure
# ---------------------------------------------------------------------------


def _hedged_preflight_json() -> str:
    data = _valid_preflight_dict()
    data["facial_calibration"]["trajectory_ambiguous_patterns"] = [
        "ML Engineer at a strong company. These MUST default to YES."
    ]
    return json.dumps(data)


def test_preflight_v2_lint_failure_retries_then_clean_generation_goes_live(
    tmp_path, monkeypatch, capsys
):
    pipeline = _make_pipeline(str(tmp_path), tmp_path)

    calls = {"n": 0}

    def _hedged_then_clean(*args, **kwargs):
        calls["n"] += 1
        return _hedged_preflight_json() if calls["n"] == 1 else _valid_preflight_json()

    monkeypatch.setattr("shared.llm_clients.opus_llm", _hedged_then_clean)

    fake_reloaded_brief = MagicMock()
    fake_reloaded_brief.has_v2_schema = False
    monkeypatch.setattr(
        "shared.brief_loader._load_v2_brief", MagicMock(return_value=fake_reloaded_brief)
    )

    with patch("linkedin.orchestrator.init_judger"):
        pipeline._run_preflight_v2()  # must not raise — retry produced a clean brief

    assert calls["n"] == 2
    captured = capsys.readouterr()
    assert "hedge_language" in captured.out


def test_preflight_v2_lint_failure_twice_aborts_hedged_brief_never_goes_live(
    tmp_path, monkeypatch
):
    pipeline = _make_pipeline(str(tmp_path), tmp_path)

    monkeypatch.setattr(
        "shared.llm_clients.opus_llm", lambda *a, **k: _hedged_preflight_json()
    )
    mock_load_v2 = MagicMock()
    monkeypatch.setattr("shared.brief_loader._load_v2_brief", mock_load_v2)

    from shared.preflight_v2 import PreflightRegimeError

    with patch("linkedin.orchestrator.init_judger"):
        with pytest.raises(PreflightRegimeError):
            pipeline._run_preflight_v2()

    # The hedged brief never went live: no reload, no written brief file.
    mock_load_v2.assert_not_called()
    assert not (pipeline.output_dir / "preflight_v2_brief.json").exists()


def test_preflight_v2_lint_blocks_hiring_company_in_positive_tier(
    tmp_path, monkeypatch
):
    pipeline = _make_pipeline(str(tmp_path), tmp_path)

    data = _valid_preflight_dict()
    data["employer_signal_rules"].append(
        {
            "tier": "strong_ai",
            "employer_patterns": ["Acme Fintech"],
            "evidence_required": "none",
            "save_on_employer_alone": True,
        }
    )
    monkeypatch.setattr(
        "shared.llm_clients.opus_llm", lambda *a, **k: json.dumps(data)
    )
    mock_load_v2 = MagicMock()
    monkeypatch.setattr("shared.brief_loader._load_v2_brief", mock_load_v2)

    from shared.preflight_v2 import PreflightRegimeError

    with patch("linkedin.orchestrator.init_judger"):
        with pytest.raises(PreflightRegimeError):
            pipeline._run_preflight_v2()

    mock_load_v2.assert_not_called()


def test_preflight_v2_threads_seed_brief_instructions_as_operator_calibration(
    tmp_path, monkeypatch
):
    """Seed-brief `instructions` entries reach the preflight prompt as the
    OPERATOR CALIBRATION block — the recruiter's answer channel for
    questions preflight itself raises in preflight_confidence_notes
    (2026-07-04 SPL run: the SPL/Senior-SPL band ambiguity had no seam to
    flow back through). Locked at the production call path, not the prompt
    helper, per the implemented-but-unlocked-wire rule."""
    pipeline = _make_pipeline(str(tmp_path), tmp_path)
    pipeline.brief_obj.instructions = [
        "Seniority calibration: target 4-10 years of relevant experience."
    ]

    captured = {}

    def fake_opus(system_prompt, user_prompt, **kwargs):
        captured["prompt"] = user_prompt
        return _valid_preflight_json()

    monkeypatch.setattr("shared.llm_clients.opus_llm", fake_opus)

    fake_reloaded_brief = MagicMock()
    fake_reloaded_brief.has_v2_schema = False
    monkeypatch.setattr(
        "shared.brief_loader._load_v2_brief",
        MagicMock(return_value=fake_reloaded_brief),
    )

    with patch("linkedin.orchestrator.init_judger"):
        pipeline._run_preflight_v2()

    assert "OPERATOR CALIBRATION" in captured["prompt"]
    assert (
        "Seniority calibration: target 4-10 years of relevant experience."
        in captured["prompt"]
    )


def test_preflight_v2_threads_seed_brief_intake_notes_into_prompt(
    tmp_path, monkeypatch
):
    """Seed-brief `intake_notes` reaches the preflight prompt as the
    RECRUITER INTAKE NOTES block — before this seam existed preflight had no
    input besides the JD, so a seed's intake never influenced brief
    generation (W0, plans/formation-prompt-de-prescribed.md). Locked at the
    production call path, not the prompt helper, per the
    implemented-but-unlocked-wire rule."""
    pipeline = _make_pipeline(str(tmp_path), tmp_path)
    pipeline.brief_obj.instructions = []
    pipeline.brief_obj.intake_notes = (
        "Priority 1: widget-calibration operations background."
    )

    captured = {}

    def fake_opus(system_prompt, user_prompt, **kwargs):
        captured["prompt"] = user_prompt
        return _valid_preflight_json()

    monkeypatch.setattr("shared.llm_clients.opus_llm", fake_opus)

    fake_reloaded_brief = MagicMock()
    fake_reloaded_brief.has_v2_schema = False
    monkeypatch.setattr(
        "shared.brief_loader._load_v2_brief",
        MagicMock(return_value=fake_reloaded_brief),
    )

    with patch("linkedin.orchestrator.init_judger"):
        pipeline._run_preflight_v2()

    assert "RECRUITER INTAKE NOTES" in captured["prompt"]
    assert (
        "Priority 1: widget-calibration operations background."
        in captured["prompt"]
    )


def test_preflight_v2_non_string_intake_notes_mean_no_intake_block(
    tmp_path, monkeypatch
):
    """Anything but a real string (including a bare MagicMock attribute, the
    default in this harness) is treated as no intake — same untrusted-shape
    rule as the instructions channel."""
    pipeline = _make_pipeline(str(tmp_path), tmp_path)
    # _make_pipeline's brief is a MagicMock: .intake_notes is an auto-Mock,
    # not a string — exactly the untrusted shape.

    captured = {}

    def fake_opus(system_prompt, user_prompt, **kwargs):
        captured["prompt"] = user_prompt
        return _valid_preflight_json()

    monkeypatch.setattr("shared.llm_clients.opus_llm", fake_opus)

    fake_reloaded_brief = MagicMock()
    fake_reloaded_brief.has_v2_schema = False
    monkeypatch.setattr(
        "shared.brief_loader._load_v2_brief",
        MagicMock(return_value=fake_reloaded_brief),
    )

    with patch("linkedin.orchestrator.init_judger"):
        pipeline._run_preflight_v2()

    assert "RECRUITER INTAKE NOTES" not in captured["prompt"]


def test_preflight_v2_non_list_instructions_mean_no_calibration_block(
    tmp_path, monkeypatch
):
    """Anything but a real list (including a bare MagicMock attribute, the
    default in this harness) is treated as no guidance — the channel only
    trusts what the loader actually produced."""
    pipeline = _make_pipeline(str(tmp_path), tmp_path)
    # _make_pipeline's brief is a MagicMock: .instructions is an auto-Mock,
    # not a list — exactly the untrusted shape.

    captured = {}

    def fake_opus(system_prompt, user_prompt, **kwargs):
        captured["prompt"] = user_prompt
        return _valid_preflight_json()

    monkeypatch.setattr("shared.llm_clients.opus_llm", fake_opus)

    fake_reloaded_brief = MagicMock()
    fake_reloaded_brief.has_v2_schema = False
    monkeypatch.setattr(
        "shared.brief_loader._load_v2_brief",
        MagicMock(return_value=fake_reloaded_brief),
    )

    with patch("linkedin.orchestrator.init_judger"):
        pipeline._run_preflight_v2()

    assert "OPERATOR CALIBRATION" not in captured["prompt"]


def test_preflight_v2_reprints_confidence_notes_last(tmp_path, monkeypatch, capsys):
    """The confidence notes are preflight's open questions for the operator;
    the full review rendering scrolls away (~80 lines before 'complete'), so
    the banner must ALSO be the last thing preflight prints (2026-07-04 SPL
    run: the SPL/Senior-SPL band question went unread)."""
    pipeline = _make_pipeline(str(tmp_path), tmp_path)

    data = _valid_preflight_dict()
    data["preflight_confidence_notes"] = "Role level is ambiguous — confirm the band."
    monkeypatch.setattr(
        "shared.llm_clients.opus_llm", lambda *a, **k: json.dumps(data)
    )

    fake_reloaded_brief = MagicMock()
    fake_reloaded_brief.has_v2_schema = False
    monkeypatch.setattr(
        "shared.brief_loader._load_v2_brief",
        MagicMock(return_value=fake_reloaded_brief),
    )

    with patch("linkedin.orchestrator.init_judger"):
        pipeline._run_preflight_v2()

    out = capsys.readouterr().out
    banner = "⚠  PREFLIGHT CONFIDENCE NOTES — REVIEW THESE CAREFULLY"
    complete = "Preflight V2 complete"
    # Rendered twice: once inside the review document, once re-printed last.
    assert out.count(banner) == 2
    assert out.rindex(banner) > out.index(complete)
    assert "Role level is ambiguous — confirm the band." in out


def test_preflight_v2_without_notes_prints_no_trailing_banner(
    tmp_path, monkeypatch, capsys
):
    pipeline = _make_pipeline(str(tmp_path), tmp_path)

    monkeypatch.setattr(
        "shared.llm_clients.opus_llm", lambda *a, **k: _valid_preflight_json()
    )
    fake_reloaded_brief = MagicMock()
    fake_reloaded_brief.has_v2_schema = False
    monkeypatch.setattr(
        "shared.brief_loader._load_v2_brief",
        MagicMock(return_value=fake_reloaded_brief),
    )

    with patch("linkedin.orchestrator.init_judger"):
        pipeline._run_preflight_v2()

    assert "PREFLIGHT CONFIDENCE NOTES" not in capsys.readouterr().out
